"""Command-line entry point: ``footymind <command>``."""

from __future__ import annotations

import argparse
import json
import sys
import time


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="footymind")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_dl = sub.add_parser("download", help="download raw provider data with checksum verification")
    p_dl.add_argument("match_ids", nargs="*", help="default: all registered sources")

    p_in = sub.add_parser("ingest", help="build processed match bundles")
    p_in.add_argument("match_ids", nargs="*", help="default: all CSV-format Metrica games")

    p_srv = sub.add_parser("serve", help="run the API server")
    p_srv.add_argument("--host", default="127.0.0.1")
    p_srv.add_argument("--port", type=int, default=8000)

    p_exp = sub.add_parser("experiment", help="run a research experiment from a config file")
    p_exp.add_argument("config", help="path to experiment JSON config")

    args = parser.parse_args(argv)

    if args.cmd == "download":
        from footymind.data import sources

        ids = args.match_ids or list(sources.SOURCES)
        for mid in ids:
            t0 = time.time()
            sources.download(sources.SOURCES[mid])
            print(f"{mid}: verified ({time.time() - t0:.1f}s)")
    elif args.cmd == "ingest":
        from footymind.data import sources, store

        ids = args.match_ids or [m for m, s in sources.SOURCES.items() if s.fmt == "metrica_csv"]
        for mid in ids:
            t0 = time.time()
            out = store.ingest(mid)
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
