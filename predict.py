# """Required offline inference entrypoint for the DLAM private evaluation."""

# from __future__ import annotations

# import argparse
# from pathlib import Path

# from src.inference import run_ensemble_inference


# def main() -> None:
#     parser = argparse.ArgumentParser(description="Run the final TFT-TCN-Chronos ensemble.")
#     parser.add_argument("--input_dir", required=True, type=Path)
#     parser.add_argument("--output_file", required=True, type=Path)
#     parser.add_argument("--checkpoint", required=True, type=Path)
#     args = parser.parse_args()

#     run_ensemble_inference(
#         input_dir=args.input_dir,
#         output_file=args.output_file,
#         checkpoint_path=args.checkpoint,
#     )


# if __name__ == "__main__":
#     main()

"""Required offline inference entrypoint for the DLAM private evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.inference import run_ensemble_inference


def resolve_path(p: Path) -> Path:
    """Return p unchanged if it exists. If it's an absolute path that doesn't
    exist (e.g. no /submission mount on a local dev box), fall back to the
    same path resolved relative to the current working directory. In the
    real eval container, the absolute path exists and this fallback never
    triggers — the required invocation is untouched."""
    if p.exists():
        return p
    if p.is_absolute():
        fallback = Path.cwd() / p.relative_to(p.anchor)
        if fallback.exists():
            print(f"[predict.py] {p} not found, using local fallback {fallback}", file=sys.stderr)
            return fallback
    return p


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the final TFT-TCN-Chronos ensemble.")
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_file", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    args = parser.parse_args()

    run_ensemble_inference(
        input_dir=resolve_path(args.input_dir),
        output_file=args.output_file if args.output_file.is_absolute() and args.output_file.parent.exists() else resolve_path(args.output_file.parent) / args.output_file.name,
        checkpoint_path=resolve_path(args.checkpoint),
    )


if __name__ == "__main__":
    main()