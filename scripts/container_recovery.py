#!/usr/bin/env python3
"""Exercise the real container entrypoint's recovery paths on a data volume. Standard library only.

    python3 scripts/container_recovery.py docker --image footymind:ci
    python3 scripts/container_recovery.py local            # runs docker/entrypoint.sh directly

Both modes run the actual ``docker/entrypoint.sh`` followed by ``footymind serve``. ``docker``
mode runs it inside the built image on a named Docker volume. ``local`` mode runs the same
script as a process against a temporary FOOTYMIND_DATA_DIR, with the installed ``footymind``
and ``python`` on PATH and the local Metrica raw files linked in, so no data is downloaded.

Scenarios (each is a fresh start of the entrypoint on the state the previous one left, plus
the stated damage; nothing is mocked and every check reads the real API):

A  first start on an empty volume, then a restart on that healthy volume: the restart must
   not re-ingest or rebuild anything and must serve the same matches.
B  an outdated bundle (manifest from pipeline version "2"): the entrypoint re-ingests that
   source, rebuilds the now-stale index, and the API serves the repaired bundle.
C  an interrupted first start: one bundle directory without a manifest (a crash mid-write)
   and no similarity index; with the default sources the entrypoint completes the missing
   ingests and builds the index.
D  an unknown source id in FOOTYMIND_SOURCES and an unreadable bundle file: the unknown id
   is logged and skipped, the server still starts, and the unreadable match answers 503
   while the other matches keep working. Then a restart on that damaged volume with the
   index deleted: the server starts and the index is built from the readable bundles.
R  (docker mode only) the volume made root-owned and the container started as root, as on
   Railway with RAILWAY_RUN_UID=0: the server must run as the unprivileged `app` user and
   the volume must end up owned by it.

Only Metrica sample games are used (1-3). Exits non-zero on the first failed check.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
ALL3 = ["metrica-sample-1", "metrica-sample-2", "metrica-sample-3"]


class Fail(AssertionError):
    pass


def check(cond: object, msg: str) -> None:
    if not cond:
        raise Fail(msg)
    print(f"    ok  {msg}")


def get(base: str, path: str, timeout: float = 60) -> tuple[int, Any]:
    try:
        with urllib.request.urlopen(base + path, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            return e.code, json.loads(body)
        except ValueError:
            return e.code, body.decode(errors="replace")


class Runner:
    """Start/stop the entrypoint + server on one data volume, and edit files in that volume."""

    port: int

    def start(self, env: dict[str, str]) -> None:
        raise NotImplementedError

    def running(self) -> bool:
        raise NotImplementedError

    def stop(self) -> str:
        """Stop this start and return its logs."""
        raise NotImplementedError

    def sh(self, script: str) -> str:
        """Run ``sh -c script`` with $D set to the data directory; return stdout."""
        raise NotImplementedError

    def cleanup(self) -> None:
        raise NotImplementedError

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def wait_healthy(self, deadline_s: float = 900) -> float:
        t0 = time.time()
        while time.time() - t0 < deadline_s:
            if not self.running():
                raise Fail("the entrypoint/server exited before becoming healthy")
            try:
                code, _ = get(self.base, "/api/health", timeout=5)
                if code == 200:
                    return time.time() - t0
            except OSError:
                pass
            time.sleep(2)
        raise Fail(f"not healthy after {deadline_s:.0f} s")


class DockerRunner(Runner):
    def __init__(self, image: str, port: int) -> None:
        self.image, self.port = image, port
        self.volume = f"fm-recovery-{os.getpid()}"
        self.name = f"fm-recovery-{os.getpid()}"
        self.user: str | None = None  # e.g. "0": run the container as root
        subprocess.run(["docker", "volume", "create", self.volume], check=True, capture_output=True)

    def start(self, env: dict[str, str]) -> None:
        args = ["docker", "run", "-d", "--name", self.name, "-p", f"127.0.0.1:{self.port}:8000"]
        args += ["-v", f"{self.volume}:/data"]
        if self.user is not None:
            args += ["--user", self.user]
        for k, v in env.items():
            args += ["-e", f"{k}={v}"]
        subprocess.run([*args, self.image], check=True, capture_output=True)

    def running(self) -> bool:
        r = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", self.name],
            capture_output=True,
            text=True,
        )
        return r.stdout.strip() == "true"

    def stop(self) -> str:
        logs = subprocess.run(["docker", "logs", self.name], capture_output=True, text=True)
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)
        return logs.stdout + logs.stderr

    def sh(self, script: str) -> str:
        r = subprocess.run(
            [
                *("docker", "run", "--rm", "-v", f"{self.volume}:/data", "-e", "D=/data"),
                *("--entrypoint", "sh", self.image, "-c", script),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return r.stdout

    def cleanup(self) -> None:
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)
        subprocess.run(["docker", "volume", "rm", "-f", self.volume], capture_output=True)


class LocalRunner(Runner):
    def __init__(self, port: int) -> None:
        self.port = port
        raw = REPO / "data" / "raw" / "metrica"
        # link the local raw files: `footymind download` verifies their checksums and skips them
        if not raw.is_dir():
            raise SystemExit(f"local mode needs {raw} (run `footymind download {' '.join(ALL3)}`)")
        self.dir = Path(tempfile.mkdtemp(prefix="fm-recovery-"))
        (self.dir / "raw").mkdir()
        (self.dir / "raw" / "metrica").symlink_to(raw)
        venv_bin = REPO / ".venv" / "bin"
        self.path = f"{venv_bin}{os.pathsep}{os.environ['PATH']}"
        self.proc: subprocess.Popen[bytes] | None = None
        self.log = self.dir / "run.log"

    def start(self, env: dict[str, str]) -> None:
        e = {k: v for k, v in os.environ.items() if k != "FOOTYMIND_SOURCES"}
        e.update(FOOTYMIND_DATA_DIR=str(self.dir), PATH=self.path, **env)
        cmd = ["sh", str(REPO / "docker" / "entrypoint.sh"), "footymind", "serve"]
        cmd += ["--host", "127.0.0.1", "--port", str(self.port)]
        self.log.write_bytes(b"")
        fh = self.log.open("ab")
        self.proc = subprocess.Popen(cmd, env=e, stdout=fh, stderr=subprocess.STDOUT, cwd=self.dir)

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self) -> str:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None
        return self.log.read_text(errors="replace")

    def sh(self, script: str) -> str:
        r = subprocess.run(
            ["sh", "-c", script],
            env={**os.environ, "D": str(self.dir)},
            capture_output=True,
            text=True,
            check=True,
        )
        return r.stdout

    def cleanup(self) -> None:
        self.stop()
        shutil.rmtree(self.dir, ignore_errors=True)


def served_ids(r: Runner) -> list[str]:
    code, body = get(r.base, "/api/matches")
    check(code == 200, f"/api/matches answers 200 (got {code})")
    return sorted(m["match_id"] for m in body)


def check_match_works(r: Runner, mid: str) -> None:
    code, poss = get(r.base, f"/api/matches/{mid}/possessions?min_passes=3")
    check(code == 200 and len(poss) > 0, f"{mid}: possessions served ({code})")
    pid = urllib.parse.quote(poss[len(poss) // 2]["possession_id"], safe="")
    code, rep = get(r.base, f"/api/possessions/{pid}/replay")
    check(code == 200 and rep["provenance"] == "observed", f"{mid}: replay of a real possession")


def check_similar(r: Runner, mid: str) -> None:
    """The index holds pass starts: query at real PASS events of mid-match possessions."""
    _, poss = get(r.base, f"/api/matches/{mid}/possessions?min_passes=3")
    code = 0
    for p in poss[len(poss) // 2 :][:5]:
        pid = urllib.parse.quote(p["possession_id"], safe="")
        _, rep = get(r.base, f"/api/possessions/{pid}/replay")
        for e in [e for e in rep["events"] if e["type"] == "PASS"][:3]:
            code, s = get(r.base, f"/api/matches/{mid}/frames/{e['start_frame']}/similar")
            if code == 200 and s.get("neighbours"):
                check(True, f"{mid}: similar moments served at a real pass ({e['start_frame']})")
                return
    raise Fail(f"{mid}: no similar moments at 15 real passes (last HTTP {code})")


def manifest_version(r: Runner, mid: str) -> str:
    out = r.sh(f'cat "$D/processed/{mid}/manifest.json"')
    return str(json.loads(out)["pipeline_version"])


def run_once(r: Runner, env: dict[str, str], label: str) -> None:
    """One start of the entrypoint until healthy (the server keeps running)."""
    print(f"  start ({label})")
    r.start(env)
    try:
        dt = r.wait_healthy()
    except Fail:
        print(r.stop()[-4000:])
        raise
    print(f"    healthy after {dt:.0f} s")


def scenario_a(r: Runner) -> None:
    print("A: first start on an empty volume, then a restart on the healthy volume")
    env = {"FOOTYMIND_SOURCES": "metrica-sample-1 metrica-sample-2"}
    run_once(r, env, "empty volume")
    check(served_ids(r) == ALL3[:2], "first start serves games 1 and 2")
    check_similar(r, "metrica-sample-1")
    first = r.stop()
    check("no processed data" in first, "first start reported the empty volume")
    before = r.sh('ls -l --time-style=+%s "$D/processed" "$D/processed/metrica-sample-1"')
    run_once(r, env, "healthy volume")
    logs = r.stop()
    check("ingesting" not in logs, "restart did not re-ingest anything")
    check("building the" not in logs, "restart did not rebuild a similarity index")
    after = r.sh('ls -l --time-style=+%s "$D/processed" "$D/processed/metrica-sample-1"')
    check(before == after, "restart left every bundle and index file untouched")
    run_once(r, env, "healthy volume, serving")
    check(served_ids(r) == ALL3[:2], "restart serves the same matches")
    check_match_works(r, "metrica-sample-2")
    check_similar(r, "metrica-sample-2")
    r.stop()


def scenario_b(r: Runner) -> None:
    print("B: a bundle from an older pipeline version")
    mid = "metrica-sample-2"
    r.sh(
        f'f="$D/processed/{mid}/manifest.json"; '
        'sed \'s/"pipeline_version": *"[^"]*"/"pipeline_version": "2"/\' "$f" > "$f.x" '
        '&& mv "$f.x" "$f"'
    )
    check(manifest_version(r, mid) == "2", f"{mid} seeded as pipeline version 2")
    env = {"FOOTYMIND_SOURCES": "metrica-sample-1 metrica-sample-2"}
    run_once(r, env, "outdated bundle")
    check(served_ids(r) == ALL3[:2], "both matches served after recovery")
    check_match_works(r, mid)
    check_similar(r, mid)
    logs = r.stop()
    check(f"ingesting outdated or missing sources: {mid}" in logs, f"entrypoint re-ingested {mid}")
    check("building the metrica similar-moments index" in logs, "stale index was rebuilt")
    check(manifest_version(r, mid) != "2", f"{mid} is at the current pipeline version")


def scenario_c(r: Runner) -> None:
    print("C: an interrupted first start (partial bundle, no index), default sources")
    r.sh(
        'rm -rf "$D/processed/metrica-sample-2" "$D/processed/similarity_metrica.parquet" && '
        'mkdir -p "$D/processed/metrica-sample-2" && '
        'printf partial > "$D/processed/metrica-sample-2/tracking.npz"'
    )
    run_once(r, {}, "partial volume")  # FOOTYMIND_SOURCES unset: the documented default
    check(served_ids(r) == ALL3, "all three default games served after recovery")
    for mid in ALL3:
        check_match_works(r, mid)
    check_similar(r, "metrica-sample-3")
    logs = r.stop()
    line = next((x for x in logs.splitlines() if "ingesting outdated or missing sources:" in x), "")
    check(
        "metrica-sample-2" in line and "metrica-sample-3" in line,
        "the entrypoint ingested the partial and the missing source",
    )
    check("building the metrica similar-moments index" in logs, "missing index was built")


def scenario_d(r: Runner) -> None:
    print("D: an unknown source id and an unreadable bundle file")
    env = {"FOOTYMIND_SOURCES": " ".join([*ALL3, "skillcorner-0000000"])}
    run_once(r, env, "unknown source")
    check(served_ids(r) == ALL3, "the registered matches are still served")
    r.sh('printf truncated > "$D/processed/metrica-sample-3/tracking.npz"')
    code, body = get(r.base, "/api/matches/metrica-sample-3/possessions")
    check(code == 503, f"unreadable bundle answers 503, not 500 ({code})")
    check("footymind ingest" in json.dumps(body), "the 503 names the fix")
    code, _ = get(r.base, "/api/health")
    check(code == 200, "health still answers 200")
    for mid in ALL3[:2]:
        check_match_works(r, mid)
    logs = r.stop()
    check("not registered sources" in logs and "skillcorner-0000000" in logs, "unknown id logged")
    check("Traceback" not in logs, "no traceback in the logs")
    # start again on the damaged volume, with the index gone: the entrypoint must still start
    # the server and build the index from the readable bundles only
    r.sh('rm -f "$D/processed/similarity_metrica.parquet"')
    run_once(r, {"FOOTYMIND_SOURCES": " ".join(ALL3)}, "damaged volume, no index")
    check(served_ids(r) == ALL3, "the damaged match stays listed (its manifest is valid)")
    code, _ = get(r.base, "/api/matches/metrica-sample-3/possessions")
    check(code == 503, f"the damaged match still answers 503 ({code})")
    check_similar(r, "metrica-sample-1")
    logs = r.stop()
    check("metrica-sample-3 could not be read; left out" in logs, "index build skipped it")
    check("Traceback" not in logs, "no traceback in the logs")


def scenario_root(r: DockerRunner) -> None:
    """Docker only: a root-owned volume and a container started as root (Railway's setup with
    RAILWAY_RUN_UID=0). The entrypoint must hand the volume to `app` and drop root."""
    print("R: started as root on a root-owned volume")
    subprocess.run(
        [
            *("docker", "run", "--rm", "--user", "0", "-v", f"{r.volume}:/data"),
            *("--entrypoint", "sh", r.image, "-c", "chown -R root:root /data && chmod 755 /data"),
        ],
        check=True,
        capture_output=True,
    )
    check(r.sh('stat -c %u "$D"').strip() == "0", "volume seeded as root-owned")
    r.user = "0"
    try:
        # the volume left by scenario D: games 1-3 (game 3's data file damaged, manifest valid)
        run_once(r, {"FOOTYMIND_SOURCES": " ".join(ALL3)}, "as root")
        status = subprocess.run(
            ["docker", "exec", r.name, "sh", "-c", "grep ^Uid /proc/1/status"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
        check(status[1:3] == ["10001", "10001"], f"the server runs as uid 10001 ({status[1:3]})")
        check(served_ids(r) == ALL3, "the matches are served")
        check_match_works(r, "metrica-sample-1")
        logs = r.stop()
        check("dropping privileges" in logs, "the entrypoint logged the privilege drop")
        check("WARNING: running as root" not in logs, "no fallback to root")
    finally:
        r.user = None
    owner = r.sh('stat -c %u "$D" "$D/processed" "$D/processed/metrica-sample-1/manifest.json"')
    check(set(owner.split()) == {"10001"}, "the volume now belongs to app (uid 10001)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=["docker", "local"])
    ap.add_argument("--image", default="footymind:ci")
    ap.add_argument("--port", type=int, default=8011)
    args = ap.parse_args(argv)
    r: Runner = (
        DockerRunner(args.image, args.port) if args.mode == "docker" else LocalRunner(args.port)
    )
    try:
        for sc in (scenario_a, scenario_b, scenario_c, scenario_d):
            sc(r)
        if isinstance(r, DockerRunner):
            scenario_root(r)
    except (Fail, KeyError, ValueError, subprocess.CalledProcessError) as e:
        print(f"RECOVERY FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        with contextlib.suppress(Exception):  # best-effort log dump
            print(r.stop()[-6000:], file=sys.stderr)
        return 1
    finally:
        r.cleanup()
    print("all recovery scenarios passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
