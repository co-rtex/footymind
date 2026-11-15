"""Command-line entry point: ``footymind <command>``."""

from __future__ import annotations

import argparse
import json
import sys
import time

GROUPS_HELP = (
    "match ids, or a group: 'metrica' (3 sample games, about 180 MB), 'skillcorner' "
    "(20 broadcast matches, about 1.8 GB) or 'all'"
)


def expand_groups(names: list[str]) -> list[str]:
    """Expand ``metrica`` / ``skillcorner`` / ``all`` into registered ids; other names are kept
    as given (unvalidated), in order and without duplicates."""
    from footymind.data import sources

    out: list[str] = []
    for n in names:
        if n == "all":
            ids = list(sources.SOURCES)
        elif n in ("metrica", "skillcorner"):
            ids = [m for m in sources.SOURCES if m.startswith(n + "-")]
        else:
            ids = [n]
        out += [m for m in ids if m not in out]
    return out


def resolve_sources(names: list[str]) -> list[str]:
    """Expand ``metrica`` / ``skillcorner`` / ``all`` into registered match ids, in order.

    Nothing is implied: an empty list is an error (a bare ``footymind download`` used to fetch
    all 23 sources, about 2 GB), and so is an unknown id.
    """
    from footymind.data import sources

    if not names:
        raise SystemExit(f"name the sources: {GROUPS_HELP}")
    out = expand_groups(names)
    unknown = [m for m in out if m not in sources.SOURCES]
    if unknown:
        raise SystemExit(f"unknown source {unknown[0]!r}: {GROUPS_HELP}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="footymind")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_dl = sub.add_parser("download", help="download raw provider data with checksum verification")
    p_dl.add_argument("match_ids", nargs="*", help=GROUPS_HELP)

    p_in = sub.add_parser("ingest", help="build processed match bundles")
    p_in.add_argument("match_ids", nargs="*", help=GROUPS_HELP + " (run `download` first)")

    p_srv = sub.add_parser("serve", help="run the API server")
    p_srv.add_argument("--host", default="127.0.0.1")
    p_srv.add_argument("--port", type=int, default=8000)

    p_exp = sub.add_parser("experiment", help="run a research experiment from a config file")
    p_exp.add_argument("config", help="path to experiment JSON config")

    p_opt = sub.add_parser("train-option-model", help="evaluate (LOMO) and export the option model")
    p_opt.add_argument("match_ids", nargs="*", help="default: all processed matches of --provider")
    p_opt.add_argument("--provider", default="metrica", choices=["metrica", "skillcorner"])

    p_sim = sub.add_parser(
        "build-similarity-index", help="build the per-provider similar-moments index"
    )
    p_sim.add_argument(
        "--provider", nargs="*", default=["metrica", "skillcorner"], help="providers to index"
    )
    p_rsp = sub.add_parser(
        "train-response-model",
        help="export the learned defender-response model (run response_validation first)",
    )
    p_rsp.add_argument("match_ids", nargs="*", help="default: all processed matches of --provider")
    p_rsp.add_argument("--provider", default="metrica", choices=["metrica", "skillcorner"])

    args = parser.parse_args(argv)

    if args.cmd == "download":
        from footymind.data import sources

        for mid in resolve_sources(args.match_ids):
            t0 = time.time()
            sources.download(sources.SOURCES[mid])
            print(f"{mid}: verified ({time.time() - t0:.1f}s)")
    elif args.cmd == "ingest":
        from footymind.data import sources, store

        for mid in resolve_sources(args.match_ids):
            t0 = time.time()
            try:
                out = store.ingest(mid)
            except FileNotFoundError as e:
                raise SystemExit(
                    f"{mid}: raw data missing ({e.filename or e}); "
                    f"run `footymind download {mid}` first"
                ) from e
            q = json.loads((out / "manifest.json").read_text())["quality"]["event_sync"]
            print(
                f"{mid}: -> {out} ({time.time() - t0:.1f}s); passer-ball median "
                f"{q['passer_to_ball_at_start_m']['median']} m"
            )
    elif args.cmd == "serve":
        import uvicorn

        uvicorn.run("footymind.api.app:app", host=args.host, port=args.port)
    elif args.cmd == "experiment":
        from footymind.research.runner import run_from_config

        run_from_config(args.config)
    elif args.cmd == "train-option-model":
        from footymind.data.store import available_matches
        from footymind.research.train_option_model import train

        ids = args.match_ids or [
            m for m in available_matches() if m.startswith(args.provider + "-")
        ]
        m = train(ids)
        print(json.dumps({k: m.meta[k] for k in ("n_train", "lomo_pooled", "code_revision")}))
    elif args.cmd == "build-similarity-index":
        from footymind.data.store import available_matches, is_servable
        from footymind.research.similarity import build_index, indexable_matches

        for prov in args.provider:
            ids, broken = indexable_matches(prov)
            stale = [
                mid
                for mid in available_matches()
                if mid.startswith(prov + "-") and not is_servable(mid)
            ]
            for mid in stale:
                print(f"{prov}: {mid} needs `footymind ingest {mid}`; left out of the index")
            for mid in broken:
                print(f"{prov}: {mid} could not be read; left out (`footymind ingest {mid}`)")
            if not ids:
                print(f"{prov}: no servable matches; skipped")
                continue
            t0 = time.time()
            out = build_index(prov, ids, skipped=broken)
            print(f"{prov}: {out} ({len(ids)} matches, {time.time() - t0:.0f}s)")
    elif args.cmd == "train-response-model":
        from footymind.research.train_response_model import train as train_response

        rm = train_response(args.provider, args.match_ids or None)
        print(json.dumps({k: rm.meta[k] for k in ("n_train", "validation", "code_revision")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
