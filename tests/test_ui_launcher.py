from __future__ import annotations

import os
import subprocess
import sys

from ids_project.ui_launcher import build_parser, build_streamlit_command


def test_ids_ui_parser_accepts_launch_arguments():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--artifact-dir",
            "artifacts/final",
            "--release-summary",
            "reports/release/summary.json",
            "--external-report",
            "reports/external_validation/default-prod.json",
            "--host",
            "0.0.0.0",
            "--port",
            "8600",
        ]
    )

    assert args.artifact_dir == "artifacts/final"
    assert args.host == "0.0.0.0"
    assert args.port == 8600


def test_ids_ui_builds_streamlit_command_without_running_server():
    args = build_parser().parse_args(["--port", "8600"])

    command = build_streamlit_command(args)

    assert command[1:4] == ["-m", "streamlit", "run"]
    assert "--server.port" in command
    assert "8600" in command
    assert "--artifact-dir" in command
    assert "artifacts/final" in command


def test_ids_ui_help_exits_successfully():
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    result = subprocess.run(
        [sys.executable, "-m", "ids_project.ui_launcher", "--help"],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode == 0
    assert "ids-ui" in result.stdout
