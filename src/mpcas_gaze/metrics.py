import math
from collections import defaultdict

import numpy as np


def euclidean_distance(pred_x, pred_y, label_x, label_y):
    return math.sqrt((pred_x - label_x) ** 2 + (pred_y - label_y) ** 2)


def frame_error(records):
    if not records:
        return float("nan")
    return float(np.mean([record["error"] for record in records]))


def dot_error(records, group="target"):
    if not records:
        return 0, float("nan")
    grouped = defaultdict(list)
    for record in records:
        if group == "dotnum":
            key = (record["subject"], record.get("dotNum", -1), record.get("orientation", -1))
        elif group == "target":
            key = (
                record["subject"],
                round(float(record["label_x"]), 6),
                round(float(record["label_y"]), 6),
                record.get("orientation", -1),
            )
        else:
            raise ValueError(f"Unsupported dot error group: {group}")
        grouped[key].append(record)

    errors = []
    for group_records in grouped.values():
        pred_x = float(np.mean([record["pred_x"] for record in group_records]))
        pred_y = float(np.mean([record["pred_y"] for record in group_records]))
        label_x = float(np.mean([record["label_x"] for record in group_records]))
        label_y = float(np.mean([record["label_y"] for record in group_records]))
        errors.append(euclidean_distance(pred_x, pred_y, label_x, label_y))
    return len(errors), float(np.mean(errors))


def summarize(records):
    _, target_dot_error = dot_error(records, group="target")
    return {
        "error": frame_error(records),
        "dot.error": target_dot_error,
    }


def summarize_by(records, key):
    summaries = {}
    preferred_order = ["phone", "tablet", "unknown"]
    values = sorted({record.get(key, "unknown") for record in records})
    ordered = [value for value in preferred_order if value in values]
    ordered.extend(value for value in values if value not in ordered)
    for value in ordered:
        summaries[value] = summarize([record for record in records if record.get(key, "unknown") == value])
    return summaries
