"""Command-line entry points for preparing data and running models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import assemble_tasks
from .experiment import run_experiments
from .fusion import fuse_modalities
from .interaction import extract_interaction


def main() -> None:
    parser = argparse.ArgumentParser(prog="mpcas")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Join task tables into the model input format")
    prepare.add_argument("--task-dir", type=Path, required=True)
    prepare.add_argument("--metadata", type=Path, required=True, help="Participant metadata CSV or sub_label.xlsx")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--config", type=Path, default=Path("configs/experiments.json"))
    run = sub.add_parser("run", help="Nested discovery CV and locked validation")
    run.add_argument("--data", type=Path, required=True)
    run.add_argument("--config", type=Path, default=Path("configs/experiments.json"))
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--profiles", nargs="+", default=["rapid_screening", "diagnostic_aid", "mci_subtyping"])
    interaction = sub.add_parser("extract-interaction", help="Extract screen-interaction metrics from response.json")
    interaction.add_argument("--data-root", type=Path, required=True)
    interaction.add_argument("--output", type=Path, required=True)
    interaction.add_argument("--tasks", nargs="+", required=True)
    interaction.add_argument("--versions", nargs="+", default=["self"])
    audio = sub.add_parser("extract-audio", help="Extract acoustic and language features from task recordings")
    audio.add_argument("--data-root", type=Path, required=True)
    audio.add_argument("--output-dir", type=Path, required=True)
    audio.add_argument("--tasks", nargs="+", default=["T004", "T009", "T013"])
    audio.add_argument("--versions", nargs="+", default=["self"])
    audio.add_argument("--phases", nargs="+", default=["formal", "practice"])
    audio.add_argument("--overwrite", action="store_true")
    fuse = sub.add_parser("fuse-features", help="Combine four modality feature summaries into task tables")
    fuse.add_argument("--metadata", type=Path, required=True)
    fuse.add_argument("--output-dir", type=Path, required=True)
    fuse.add_argument("--tasks", nargs="+", required=True)
    for name in ["screen-interaction", "gaze", "facial", "audio-language"]:
        fuse.add_argument(f"--{name}", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        config = json.loads(args.config.read_text(encoding="utf-8"))
        tasks = sorted({task for profile in config["profiles"].values() for task in profile["tasks"]})
        assemble_tasks(args.task_dir, args.metadata, args.output, tasks)
    elif args.command == "run":
        run_experiments(args.data, args.config, args.output, args.profiles)
    elif args.command == "extract-interaction":
        extract_interaction(args.data_root, args.output, args.tasks, args.versions)
    elif args.command == "extract-audio":
        from mpcas_tasks.audio_extraction import extract_audio_features

        extract_audio_features(
            data_root=args.data_root,
            output_dir=args.output_dir,
            tasks=args.tasks,
            target_tasks=args.tasks,
            phases=args.phases,
            versions=args.versions,
            overwrite=args.overwrite,
        )
    else:
        inputs = {
            name: getattr(args, name.replace("-", "_"))
            for name in ["screen-interaction", "gaze", "facial", "audio-language"]
            if getattr(args, name.replace("-", "_")) is not None
        }
        fuse_modalities(args.metadata, args.output_dir, args.tasks, inputs)


if __name__ == "__main__":
    main()
