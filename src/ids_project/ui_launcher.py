from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ids-ui")
    parser.add_argument("--artifact-dir", default="artifacts/final")
    parser.add_argument("--release-summary", default="reports/release/summary.json")
    parser.add_argument("--external-report", default="reports/external_validation/default-prod.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8501)
    return parser


def build_streamlit_command(args: argparse.Namespace) -> list[str]:
    app_path = Path(__file__).resolve().parent / "ui" / "app.py"
    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        args.host,
        "--server.port",
        str(args.port),
        "--",
        "--artifact-dir",
        args.artifact_dir,
        "--release-summary",
        args.release_summary,
        "--external-report",
        args.external_report,
    ]


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    command = build_streamlit_command(args)
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
