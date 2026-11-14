#!/usr/bin/env python3
"""Smoke-check a running FootyMind server (API + web lab) over HTTP. Standard library only.

    python3 scripts/smoke_api.py http://127.0.0.1:8000 [--wait 600] [--expect-matches a b ...]
                                                         [--require-similar] [--no-web]

Every check reads real responses from the server; nothing is mocked. For each provider served it
checks the possession replay, the frame analysis (explanations, option model, temporal context),
the response policies, a scenario with a response policy and, when the provider has at least two
matches, the similar-moments endpoint with its validated estimate and validation summary. Every
JSON body is parsed strictly: NaN or Infinity anywhere fails the run. Exits non-zero on the first
failure.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Any

PROVENANCE = {"observed", "derived", "estimated", "hypothetical", "simulated"}
POLICIES = {"none", "marker_follows", "press_carrier", "zonal_shift", "lane_cover", "learned"}


class SmokeError(AssertionError):
    pass


def check(cond: object, msg: str) -> None:
    if not cond:
        raise SmokeError(msg)


def _no_constants(name: str) -> Any:
    raise SmokeError(f"non-finite JSON constant {name}")


class Client:
    def __init__(self, base: str) -> None:
        self.base = base.rstrip("/")

    def raw(self, path: str, body: Any = None, timeout: float = 120) -> tuple[int, str, bytes]:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(
            self.base + path,
            data=data,
            headers={"content-type": "application/json"} if data else {},
            method="POST" if data else "GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, r.headers.get("content-type", ""), r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.headers.get("content-type", ""), e.read()

    def json(self, path: str, body: Any = None, status: int = 200) -> Any:
        t0 = time.perf_counter()
        code, _, raw = self.raw(path, body)
        dt = (time.perf_counter() - t0) * 1000
        check(code == status, f"{path}: HTTP {code} (expected {status}): {raw[:300]!r}")
        out = json.loads(raw, parse_constant=_no_constants)
        print(f"  ok {code} {path} ({dt:.0f} ms)")
        return out


def wait_healthy(c: Client, wait_s: float) -> dict[str, Any]:
    deadline = time.time() + wait_s
    last = ""
    while time.time() < deadline:
        try:
            code, _, raw = c.raw("/api/health", timeout=5)
            if code == 200:
                h: dict[str, Any] = json.loads(raw, parse_constant=_no_constants)
                return h
            last = f"HTTP {code}"
        except (OSError, urllib.error.URLError) as e:
            last = str(e)
        time.sleep(3)
    raise SmokeError(f"server not healthy after {wait_s:.0f} s ({last})")


def pick_analysis(c: Client, match_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """A possession replay and a frame analysis at one of its passes with a carrier."""
    poss = c.json(f"/api/matches/{match_id}/possessions?min_passes=3")
    check(len(poss) > 0, f"{match_id}: no possessions with >= 3 passes")
    for p in poss[len(poss) // 2 :][:10]:  # mid-match, not the kick-off
        pid = urllib.request.quote(p["possession_id"], safe="")
        rep = c.json(f"/api/possessions/{pid}/replay")
        nf, npl = len(rep["frame"]), len(rep["players"])
        check(len(rep["players_xy"]) == nf * npl * 2, "replay players_xy has the wrong length")
        check(len(rep["ball_xy"]) == nf * 2, "replay ball_xy has the wrong length")
        check(rep["provenance"] == "observed", "replay provenance is not 'observed'")
        passes = [e for e in rep["events"] if e["type"] == "PASS"]
        for e in passes[:4]:
            a = c.json(f"/api/matches/{match_id}/frames/{e['start_frame']}/analysis")
            if a["carrier_id"] and a["pass_options"]:
                return rep, a
    raise SmokeError(f"{match_id}: no pass frame with a carrier and options in 10 possessions")


def check_analysis(a: dict[str, Any], provider: str) -> None:
    pc = a["pitch_control"]
    check(pc and len(pc["values"]) == pc["nx"] * pc["ny"], "pitch control grid size mismatch")
    check(pc["provenance"] == "estimated", "pitch control is not labelled 'estimated'")
    check(a["ball_provenance"] in PROVENANCE, "ball provenance label is invalid")
    for p in a["players"]:
        check(p["position_provenance"] in PROVENANCE, "player position provenance is invalid")
    check(a["explanations"], "frame analysis has no explanations")
    for s in a["explanations"]:
        check(s["provenance"] in PROVENANCE and s["metric"], f"explanation without provenance: {s}")
    check(a["temporal"] is not None, "frame analysis has no temporal context")
    om = a["option_model"]
    check(om is not None, f"{provider}: no option model served")
    check(om.get("provider") in (None, provider), f"{provider}: served another provider's model")
    check(
        all(o["p_success_model"] is not None for o in a["pass_options"]),
        "an option has no option-model probability",
    )


def check_scenario(c: Client, a: dict[str, Any]) -> None:
    mid, frame = a["match_id"], a["frame"]
    pol = c.json(f"/api/response-policies?match_id={mid}")
    check({p["name"] for p in pol} == POLICIES, f"unexpected response policies {pol}")
    att = a["attacking_team"]
    movers = [
        p
        for p in a["players"]
        if p["team"] == att and p["player_id"] != a["carrier_id"] and p["x"] is not None
    ]
    check(movers, "no attacker to move")
    # The marker of a moved attacker follows: try up to four attackers until a defender responds.
    for mover in movers[:4]:
        x = max(-50.0, min(50.0, mover["x"] + 3.0))
        req = {
            "match_id": mid,
            "frame": frame,
            "interventions": [{"player_id": mover["player_id"], "x": x, "y": mover["y"]}],
            "response_policy": "marker_follows",
            "ensemble": 8,
        }
        r = c.json("/api/scenarios", req)
        if any(m["provenance"] == "simulated" for m in r["movements"]):
            break
    check(
        any(m["provenance"] == "simulated" for m in r["movements"]),
        "marker_follows moved no defender for four attackers",
    )
    check(r["policy"]["name"] == "marker_follows", "scenario did not apply the response policy")
    check(r["hypothetical"]["frame"] == frame and r["observed"]["frame"] == frame, "frame mismatch")
    check(r["option_deltas"], "scenario returned no option deltas")
    check(r["disclaimer"] and r["assumptions"], "scenario without disclaimer or assumptions")
    moved = [m for m in r["movements"] if m["player_id"] == mover["player_id"]]
    check(moved and moved[0]["provenance"] == "hypothetical", "user move not 'hypothetical'")
    check(
        all(m["provenance"] in {"hypothetical", "simulated"} for m in r["movements"]),
        "a movement has an invalid provenance",
    )
    print(
        f"  scenario {mid}@{frame}: {len(r['movements'])} movements "
        f"({sum(m['provenance'] == 'simulated' for m in r['movements'])} simulated), "
        f"{r['ensemble_size']} ensemble members"
    )


def check_similar(c: Client, a: dict[str, Any], provider: str) -> None:
    mid = a["match_id"]
    s = c.json(f"/api/matches/{mid}/frames/{a['frame']}/similar?k=8")
    check(s["provider"] == provider and s["k"] == len(s["neighbours"]) > 0, "no neighbours")
    check(all(n["match_id"] != mid for n in s["neighbours"]), "a neighbour is from the same match")
    check(
        all(n["match_id"].startswith(provider + "-") for n in s["neighbours"]),
        "a neighbour is from another provider",
    )
    est = s["estimate"]
    check(est is not None and est["k"] == 50, f"{provider}: validated k = 50 estimate missing")
    check(0 < est["completion"] < 1 and 0 < est["base_rate"] < 1, "estimate out of range")
    v = s["validation"]
    check(v is not None and v["ref"].endswith("results.json"), f"{provider}: validation missing")
    check(isinstance(v["knn_beats_base_rate"], bool), "validation verdict is not a boolean")


def check_web(c: Client) -> None:
    code, ctype, body = c.raw("/")
    check(code == 200 and "text/html" in ctype, f"/: HTTP {code} {ctype}")
    html = body.decode()
    check('id="root"' in html, "index.html has no #root element")
    js = re.findall(r'src="(/assets/[^"]+\.js)"', html)
    check(js, "index.html references no JS bundle")
    code, ctype, body = c.raw(js[0])
    check(code == 200 and "javascript" in ctype and len(body) > 10_000, f"{js[0]}: {code} {ctype}")
    print(f"  ok 200 / and {js[0]} ({len(body) // 1024} KiB)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("base_url")
    ap.add_argument("--wait", type=float, default=0, help="seconds to wait for /api/health")
    ap.add_argument("--expect-matches", nargs="*", default=None)
    ap.add_argument(
        "--require-similar",
        action="store_true",
        help="fail if a provider with >= 2 matches has no similarity index",
    )
    ap.add_argument("--no-web", action="store_true", help="skip the frontend check")
    args = ap.parse_args(argv)
    c = Client(args.base_url)
    try:
        h = wait_healthy(c, args.wait) if args.wait else c.json("/api/health")
        check(h["status"] == "ok", f"health: {h}")
        matches = c.json("/api/matches")
        ids = [m["match_id"] for m in matches]
        check(ids == h["matches"], "health and /api/matches disagree")
        if args.expect_matches is not None:
            check(set(ids) == set(args.expect_matches), f"served {ids}, expected the given set")
        check(ids, "no matches served")
        c.json("/api/matches/no-such-match", status=404)
        by_provider: dict[str, list[str]] = {}
        for m in matches:
            by_provider.setdefault(m["source"]["provider"], []).append(m["match_id"])
        for provider, mids in sorted(by_provider.items()):
            print(f"{provider}: {len(mids)} matches")
            _, a = pick_analysis(c, mids[0])
            check_analysis(a, provider)
            check_scenario(c, a)
            if len(mids) >= 2:
                code, _, raw = c.raw(f"/api/matches/{mids[0]}/frames/{a['frame']}/similar")
                if code == 503 and not args.require_similar:
                    print(f"  skip similar ({provider}): no index ({raw[:120]!r})")
                else:
                    check_similar(c, a, provider)
        if not args.no_web:
            check_web(c)
    except (SmokeError, KeyError, StopIteration, json.JSONDecodeError) as e:
        print(f"SMOKE FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print("smoke checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
