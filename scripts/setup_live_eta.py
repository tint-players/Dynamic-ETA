from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESTINATION = REPO_ROOT / "artifacts" / "eta_winner.pt"
RUN_ID = "34871556440"
ARTIFACT_NAME = "eta-experiment-20runs-12epochs-2s"
EXPECTED_SHA256 = "24feb1ca5cc30d84698a331324588e4305dcb4915dee77dc4680a11507b18655"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    actual = sha256(path)
    if actual != EXPECTED_SHA256:
        raise RuntimeError(
            f"checkpoint SHA256 mismatch for {path}: expected {EXPECTED_SHA256}, got {actual}"
        )


def download_with_gh(destination: Path) -> None:
    gh = shutil.which("gh")
    if gh is None:
        raise RuntimeError(
            "GitHub CLI is not installed. Install/authenticate `gh`, or copy the checkpoint "
            "manually and rerun this script with --source."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            gh,
            "run",
            "download",
            RUN_ID,
            "-R",
            "tint-players/Dynamic-ETA",
            "-n",
            ARTIFACT_NAME,
            "-D",
            str(destination.parent),
        ],
        check=True,
    )
    downloaded = destination.parent / "eta_winner.pt"
    if downloaded != destination:
        shutil.move(str(downloaded), str(destination))


def main() -> int:
    parser = argparse.ArgumentParser(description="Install the selected live hybrid ETA checkpoint")
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--source", type=Path, help="Copy a local eta_winner.pt instead of downloading")
    parser.add_argument("--force", action="store_true", help="Replace an existing checkpoint")
    args = parser.parse_args()

    destination = args.destination.expanduser().resolve()
    if destination.exists() and not args.force:
        verify(destination)
        print(f"Live ETA checkpoint already installed and verified: {destination}")
        return 0

    if args.source is not None:
        source = args.source.expanduser().resolve()
        verify(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    else:
        download_with_gh(destination)

    verify(destination)
    print(f"Installed verified live ETA checkpoint: {destination}")
    print(f"SHA256: {EXPECTED_SHA256}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Live ETA setup failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
