from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    root = Path(__file__).resolve().parent
    src = root / "src"
    if src.exists():
        sys.path.insert(0, str(src))

    try:
        from ids_project.cli import main as cli_main
    except ModuleNotFoundError as e:
        if e.name == "tqdm":
            subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
            from ids_project.cli import main as cli_main
        else:
            raise

    cli_main(["train", *(argv if argv is not None else sys.argv[1:])])


if __name__ == "__main__":
    main()