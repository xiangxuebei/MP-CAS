import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
import torch
import yaml

try:
    from .calibration import build_calibrator, normalize_support_tasks, select_support_indices
    from .dataset import image_batch_to_float, make_dataloader
    from .filters import apply_filter
    from .metrics import summarize, summarize_by
    from .model import build_model_from_config, strip_module_prefix
except ImportError:  # Allow direct script execution.
    from calibration import build_calibrator, normalize_support_tasks, select_support_indices
    from dataset import image_batch_to_float, make_dataloader
    from filters import apply_filter
    from metrics import summarize, summarize_by
    from model import build_model_from_config, strip_module_prefix


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CoGazeNet.")
    parser.add_argument("--config", default="configs/mpcas_gaze_example.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--device-type",
        default=None,
        choices=["phone", "tablet", "unknown"],
        help="Filter test samples by classified device type.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--loader-threads",
        type=int,
        default=None,
        help="Threads used to load/decode samples inside each batch (keeps num_workers=0).",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--calibration-mode", default=None, choices=["none", "offset", "affine", "ridge", "svr", "mvo-svr"])
    parser.add_argument("--calibration-points", type=int, default=None)
    parser.add_argument(
        "--calibration-support-tasks",
        default=None,
        help="Comma-separated task ids allowed as calibration support, e.g. T002.",
    )
    parser.add_argument("--filter", default=None, choices=["none", "heuristic", "one_euro", "kalman"])
    parser.add_argument(
        "--predictions-only",
        action="store_true",
        help=(
            "Stream predictions.csv directly to disk without retaining records, "
            "computing metrics, or writing summary.json. Requires calibration/filter mode none."
        ),
    )
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as infile:
        return yaml.safe_load(infile) or {}


def override_config(config, args):
    if args.data_path:
        config.setdefault("data", {}).setdefault("test", {})["path"] = args.data_path
    if args.batch_size:
        config.setdefault("data", {})["batch_size"] = args.batch_size
    if args.loader_threads is not None:
        config.setdefault("data", {})["loader_threads"] = max(1, args.loader_threads)
    if args.device_type:
        data_config = config.setdefault("data", {})
        data_config["device_type_filter"] = args.device_type
        data_config.setdefault("test", {})["device_type_filter"] = args.device_type
    if args.max_samples:
        config.setdefault("data", {}).setdefault("test", {})["max_samples"] = args.max_samples
    if args.save_dir:
        config.setdefault("test", {})["save_dir"] = args.save_dir
    if args.calibration_mode is not None:
        config.setdefault("calibration", {})["mode"] = args.calibration_mode
    if args.calibration_points is not None:
        config.setdefault("calibration", {})["points"] = args.calibration_points
    if args.calibration_support_tasks is not None:
        config.setdefault("calibration", {})["support_tasks"] = args.calibration_support_tasks
    if args.filter is not None:
        config.setdefault("filter", {})["mode"] = args.filter
    return config


def load_model(config, checkpoint, device):
    model = build_model_from_config(config)
    payload = torch.load(checkpoint, map_location=device)
    model.load_state_dict(strip_module_prefix(payload), strict=True)
    model.to(device)
    model.eval()
    return model


def collect_records(model, loader, device):
    records = []
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            face = image_batch_to_float(data["faceImg"], device, non_blocking=True)
            left = image_batch_to_float(data["leftEyeImg"], device, non_blocking=True)
            right = image_batch_to_float(data["rightEyeImg"], device, non_blocking=True)
            rects = data["rects"].to(device, non_blocking=True)
            aux = data["aux"].to(device, non_blocking=True)
            output = model(left, right, face, rects, aux=aux, return_features=True)
            pred = output["gaze"].cpu().numpy()
            feature = output["feature"].cpu().numpy()
            labels = data["label"].numpy()
            errors = np.linalg.norm(pred - labels, axis=1)

            for idx in range(pred.shape[0]):
                records.append(
                    {
                        "subject": data["subject"][idx],
                        "frame": data["frame"][idx],
                        "frameIndex": int(data["frameIndex"][idx].item()),
                        "dotNum": int(data["dotNum"][idx].item()),
                        "orientation": int(data["orientation"][idx].item()),
                        "deviceType": data["deviceType"][idx],
                        "deviceName": data["deviceName"][idx],
                        "task": data.get("task", [""] * pred.shape[0])[idx],
                        "phase": data.get("phase", [""] * pred.shape[0])[idx],
                        "attValid": int(data["attValid"][idx].item()) if "attValid" in data else -1,
                        "pred_x": float(pred[idx, 0]),
                        "pred_y": float(pred[idx, 1]),
                        "raw_pred_x": float(pred[idx, 0]),
                        "raw_pred_y": float(pred[idx, 1]),
                        "label_x": float(labels[idx, 0]),
                        "label_y": float(labels[idx, 1]),
                        "error": float(errors[idx]),
                        "feature": feature[idx],
                        "is_calibration": False,
                    }
                )
            print(f"\r[Batch {batch_idx + 1}/{len(loader)}]", end="")
    print()
    return records


PREDICTION_COLUMNS = [
    "subject",
    "frame",
    "frameIndex",
    "dotNum",
    "orientation",
    "deviceType",
    "deviceName",
    "task",
    "phase",
    "attValid",
    "is_calibration",
    "pred_x",
    "pred_y",
    "raw_pred_x",
    "raw_pred_y",
    "label_x",
    "label_y",
    "error",
]


def stream_predictions(model, loader, device, save_dir):
    """Write predictions batch by batch with memory bounded by one batch."""
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "predictions.csv")
    partial_path = csv_path + ".partial"

    with open(partial_path, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(PREDICTION_COLUMNS)
        with torch.inference_mode():
            for batch_idx, data in enumerate(loader):
                face = image_batch_to_float(data["faceImg"], device, non_blocking=True)
                left = image_batch_to_float(data["leftEyeImg"], device, non_blocking=True)
                right = image_batch_to_float(data["rightEyeImg"], device, non_blocking=True)
                rects = data["rects"].to(device, non_blocking=True)
                aux = data["aux"].to(device, non_blocking=True)
                pred = model(left, right, face, rects, aux=aux, return_features=False).cpu().numpy()
                labels = data["label"].numpy()
                errors = np.linalg.norm(pred - labels, axis=1)
                tasks = data.get("task", [""] * pred.shape[0])
                phases = data.get("phase", [""] * pred.shape[0])

                rows = []
                for idx in range(pred.shape[0]):
                    rows.append(
                        [
                            data["subject"][idx],
                            data["frame"][idx],
                            int(data["frameIndex"][idx].item()),
                            int(data["dotNum"][idx].item()),
                            int(data["orientation"][idx].item()),
                            data["deviceType"][idx],
                            data["deviceName"][idx],
                            tasks[idx],
                            phases[idx],
                            int(data["attValid"][idx].item()) if "attValid" in data else -1,
                            0,
                            float(pred[idx, 0]),
                            float(pred[idx, 1]),
                            float(pred[idx, 0]),
                            float(pred[idx, 1]),
                            float(labels[idx, 0]),
                            float(labels[idx, 1]),
                            float(errors[idx]),
                        ]
                    )
                writer.writerows(rows)
                print(f"\r[Batch {batch_idx + 1}/{len(loader)}]", end="", flush=True)

    print()
    os.replace(partial_path, csv_path)
    return csv_path


def calibrate_records(records, config):
    calibration_config = config.get("calibration", {})
    mode = calibration_config.get("mode", "none")
    points = int(calibration_config.get("points", 0))
    support_tasks = normalize_support_tasks(calibration_config.get("support_tasks"))
    if mode in {"none", None} or points <= 0:
        return records, {"mode": "none", "points": 0, "subjects": {}}

    output = []
    report = {"mode": mode, "points": points, "support_tasks": sorted(support_tasks) if support_tasks else None, "subjects": {}}
    grouped = defaultdict(list)
    for record in records:
        grouped[record["subject"]].append(record)

    for subject, subject_records in grouped.items():
        subject_records = sorted(subject_records, key=lambda r: (r.get("frameIndex", 0), r.get("frame", "")))
        support_mask = select_support_indices(
            subject_records,
            points=points,
            strategy=calibration_config.get("strategy", "first_unique"),
            support_tasks=support_tasks,
        )
        support = [record for record, keep in zip(subject_records, support_mask) if keep]
        query = [record for record, keep in zip(subject_records, support_mask) if not keep]
        if not support or not query:
            output.extend(subject_records)
            report["subjects"][subject] = {"support": len(support), "query": len(query), "status": "skipped"}
            continue

        support_pred = np.array([[r["raw_pred_x"], r["raw_pred_y"]] for r in support], dtype=np.float32)
        support_y = np.array([[r["label_x"], r["label_y"]] for r in support], dtype=np.float32)
        support_feature = np.stack([r["feature"] for r in support]).astype(np.float32)
        query_pred = np.array([[r["raw_pred_x"], r["raw_pred_y"]] for r in query], dtype=np.float32)
        query_feature = np.stack([r["feature"] for r in query]).astype(np.float32)

        calibrator = build_calibrator(mode)
        if mode in {"offset", "affine"}:
            calibrator.fit(support_pred, support_y)
            calibrated = calibrator.predict(query_pred)
        else:
            calibrator.fit(support_pred, support_feature, support_y)
            calibrated = calibrator.predict(query_pred, query_feature)

        for record in support:
            record = dict(record)
            record["is_calibration"] = True
            output.append(record)
        for record, pred in zip(query, calibrated):
            record = dict(record)
            record["pred_x"] = float(pred[0])
            record["pred_y"] = float(pred[1])
            record["error"] = float(np.linalg.norm(pred - np.array([record["label_x"], record["label_y"]])))
            output.append(record)
        report["subjects"][subject] = {"support": len(support), "query": len(query), "status": "ok"}
    return output, report


def filter_records(records, config):
    filter_config = config.get("filter", {})
    mode = filter_config.get("mode", "none")
    if mode in {"none", None}:
        return records
    output = []
    grouped = defaultdict(list)
    for record in records:
        grouped[record["subject"]].append(record)
    kwargs = {k: v for k, v in filter_config.items() if k != "mode"}
    for _, subject_records in grouped.items():
        subject_records = sorted(subject_records, key=lambda r: (r.get("frameIndex", 0), r.get("frame", "")))
        points = np.array([[r["pred_x"], r["pred_y"]] for r in subject_records], dtype=np.float32)
        filtered = apply_filter(points, mode=mode, **kwargs)
        for record, pred in zip(subject_records, filtered):
            record = dict(record)
            record["pred_x"] = float(pred[0])
            record["pred_y"] = float(pred[1])
            record["error"] = float(np.linalg.norm(pred - np.array([record["label_x"], record["label_y"]])))
            output.append(record)
    return output


def write_outputs(records, summary, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "predictions.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(PREDICTION_COLUMNS)
        for record in records:
            writer.writerow(
                [
                    record["subject"],
                    record["frame"],
                    record["frameIndex"],
                    record["dotNum"],
                    record["orientation"],
                    record["deviceType"],
                    record["deviceName"],
                    record.get("task", ""),
                    record.get("phase", ""),
                    record.get("attValid", -1),
                    int(record["is_calibration"]),
                    record["pred_x"],
                    record["pred_y"],
                    record["raw_pred_x"],
                    record["raw_pred_y"],
                    record["label_x"],
                    record["label_y"],
                    record["error"],
                ]
            )

    summary_path = os.path.join(save_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as outfile:
        json.dump(summary, outfile, indent=2)
        outfile.write("\n")
    return csv_path, summary_path


def main():
    args = parse_args()
    config = override_config(load_config(args.config), args)
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    loader = make_dataloader(config, "test", shuffle=False)
    model = load_model(config, args.checkpoint, device)

    if args.predictions_only:
        calibration_config = config.get("calibration", {})
        calibration_mode = calibration_config.get("mode", "none")
        calibration_points = int(calibration_config.get("points", 0))
        filter_mode = config.get("filter", {}).get("mode", "none")
        if calibration_mode not in {"none", None} and calibration_points > 0:
            raise ValueError("--predictions-only requires --calibration-mode none or zero calibration points")
        if filter_mode not in {"none", None}:
            raise ValueError("--predictions-only requires --filter none")
        save_dir = config.get("test", {}).get("save_dir", "outputs/mpcas_gaze/test")
        csv_path = stream_predictions(model, loader, device, save_dir)
        print(f"Saved CSV: {csv_path}")
        return

    records = collect_records(model, loader, device)
    records, calibration_report = calibrate_records(records, config)
    records = filter_records(records, config)

    eval_records = records
    if config.get("calibration", {}).get("exclude_support_from_metrics", True):
        eval_records = [record for record in records if not record.get("is_calibration", False)]
    overall = summarize(eval_records)
    summary = dict(overall)
    summary["overall"] = overall
    summary["by_subject"] = summarize_by(eval_records, "subject")
    summary["by_device_type"] = summarize_by(eval_records, "deviceType")
    summary["by_task"] = summarize_by(eval_records, "task")
    summary["by_phase"] = summarize_by(eval_records, "phase")
    summary["calibration"] = calibration_report
    summary["checkpoint"] = args.checkpoint
    summary["num_prediction_rows"] = len(records)
    summary["num_eval_rows"] = len(eval_records)

    save_dir = config.get("test", {}).get("save_dir", "outputs/mpcas_gaze/test")
    csv_path, summary_path = write_outputs(records, summary, save_dir)
    print(json.dumps(overall, indent=2))
    print(f"Saved CSV: {csv_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
