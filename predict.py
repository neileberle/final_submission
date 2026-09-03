"""Required offline inference entrypoint for the DLAM private evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.inference import run_ensemble_inference


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the final TFT-TCN-Chronos ensemble.")
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_file", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    args = parser.parse_args()

    run_ensemble_inference(
        input_dir=args.input_dir,
        output_file=args.output_file,
        checkpoint_path=args.checkpoint,
    )


if __name__ == "__main__":
    main()

