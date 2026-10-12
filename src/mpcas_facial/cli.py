"""Command line interface for the facial feature pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import aggregate_manifest, infer_task_videos


def main() -> None:
    parser = argparse.ArgumentParser(prog="mpcas-facial")
    commands = parser.add_subparsers(dest="command", required=True)
    infer = commands.add_parser("infer", help="OpenFace AU and ResMaskNet emotion signals from 30 fps videos")
    infer.add_argument("--dataset-root", type=Path, required=True)
    infer.add_argument("--output-dir", type=Path, required=True)
    infer.add_argument("--tasks", nargs="+", default=None)
    infer.add_argument("--device", default="cpu")
    infer.add_argument("--batch-size", type=int, default=64)
    infer.add_argument("--fps", type=float, default=30.0)
    infer.add_argument("--openface-exe", type=Path, default=None)
    infer.add_argument("--save-frames", action="store_true")
    infer.add_argument("--allow-missing-signals", action="store_true", help="Write available features and list missing inputs in the report")
    aggregate = commands.add_parser("aggregate", help="Task features from audited frame signal CSVs")
    aggregate.add_argument("--manifest", type=Path, required=True)
    aggregate.add_argument("--output-dir", type=Path, required=True)
    aggregate.add_argument("--fps", type=float, default=30.0)
    aggregate.add_argument("--allow-missing-signals", action="store_true", help="Write available features and list missing inputs in the report")
    args = parser.parse_args()
    if args.command == "infer":
        infer_task_videos(args.dataset_root, args.output_dir, tasks=set(args.tasks) if args.tasks else None,
                          device=args.device, batch_size=args.batch_size, fps=args.fps,
                          openface_exe=args.openface_exe, require_complete_signals=not args.allow_missing_signals,
                          save_frames=args.save_frames)
    else:
        aggregate_manifest(args.manifest, args.output_dir, fps=args.fps,
                           require_complete_signals=not args.allow_missing_signals)


if __name__ == "__main__":
    main()
