"""Required offline inference entrypoint for the DLAM private evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.inference import run_ensemble_inference
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def resolve_input(p: Path) -> Path:
    """Return ``p`` unchanged if it exists.

    If it is an absolute path that does not exist (e.g. no ``/submission``
    mount on a local dev box), fall back to the same path resolved relative to
    the current working directory, but only when that fallback actually
    exists. In the real eval container the absolute path is mounted and this
    fallback never triggers, so the required invocation is untouched.
    """
    if p.exists():
        return p
    if p.is_absolute():
        fallback = Path.cwd() / p.relative_to(p.anchor)
        if fallback.exists():
            print(
                f"[predict.py] {p} not found, using local fallback {fallback}",
                file=sys.stderr,
            )
            return fallback
    return p


def resolve_output(p: Path) -> Path:
    """Resolve the output path.

    An output path is *created*, never discovered, so it must not reuse the
    input fallback: if ``/output`` happened not to exist yet while a stale
    ``./output`` directory did, that fallback would silently write the
    predictions where the grader never looks. Honour the requested absolute
    path and create its parent; fall back only if the filesystem refuses.
    """
    if not p.is_absolute():
        return p
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    except OSError as exc:  # read-only or otherwise non-writable mount
        fallback = Path.cwd() / p.relative_to(p.anchor)
        print(
            f"[predict.py] cannot create {p.parent} ({exc}); writing to {fallback}",
            file=sys.stderr,
        )
        fallback.parent.mkdir(parents=True, exist_ok=True)
        return fallback


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the final TFT-TCN-Chronos ensemble.")
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_file", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    args = parser.parse_args()

    run_ensemble_inference(
        input_dir=resolve_input(args.input_dir),
        output_file=resolve_output(args.output_file),
        checkpoint_path=resolve_input(args.checkpoint),
    )


if __name__ == "__main__":
    main()
