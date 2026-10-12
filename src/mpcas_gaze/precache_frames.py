import argparse
import json
import os
import time
from pathlib import Path

import yaml

try:
    import lmdb
except ImportError:  # pragma: no cover
    lmdb = None

try:
    from .dataset import (
        FRAME_CACHE_MANIFEST,
        CoGazeDataset,
        crop_cache_key,
    )
except ImportError:  # Allow direct script execution.
    from dataset import (
        FRAME_CACHE_MANIFEST,
        CoGazeDataset,
        crop_cache_key,
    )


SPLIT_NAMES = ("train", "val", "test")


def parse_args():
    parser = argparse.ArgumentParser(description="Pack original frame JPG files into split LMDB caches.")
    parser.add_argument("--config", default="configs/mpcas_gaze_example.yaml")
    parser.add_argument("--split", default="all", choices=["all", *SPLIT_NAMES])
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--data-path", default=None)
    parser.add_argument(
        "--device-type",
        default=None,
        choices=["phone", "tablet", "unknown"],
        help="Filter cached samples by classified device type.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--lmdb-map-size-gb", type=float, default=None)
    parser.add_argument("--lmdb-commit-interval", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--log-interval", type=int, default=5000)
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as infile:
        return yaml.safe_load(infile) or {}


def split_params(config, split_name, args):
    data_config = config.get("data", {})
    split_config = data_config.get(split_name, {})
    common = {k: v for k, v in data_config.items() if k not in {"train", "val", "test"}}
    params = {**common, **split_config}
    if args.data_path:
        params["path"] = args.data_path
    if args.device_type:
        params["device_type_filter"] = args.device_type
    if args.max_samples is not None:
        params["max_samples"] = args.max_samples
    return params


def build_dataset(config, split_name, args):
    params = split_params(config, split_name, args)
    return CoGazeDataset(
        data_paths=params.get("path") or params.get("paths"),
        split=params.get("split", split_name),
        annotation_mode=params.get("annotation_mode", "auto"),
        label_path=params.get("label_path"),
        subject_list=params.get("subject_list"),
        device_filter=params.get("device_filter"),
        device_type_filter=params.get("device_type_filter"),
        orientation_filter=params.get("orientation_filter"),
        max_samples=params.get("max_samples"),
        iphone_only=bool(params.get("iphone_only", False)),
        aux_path=None,
        aux_dim=0,
        augment=False,
        brightness=0.0,
        contrast=0.0,
        crop_cache_dir=None,
        frame_cache_dir=None,
    )


def lmdb_map_size_bytes(dataset, override_gb=None):
    if override_gb is not None:
        return int(float(override_gb) * (1024 ** 3))
    total_bytes = 0
    missing = 0
    for item in dataset.lines:
        try:
            total_bytes += os.path.getsize(item["frame_path"])
        except OSError:
            missing += 1
    if missing:
        print(f"[FrameCache]: map size estimate skipped {missing} missing files")
    return int(max(1 * (1024 ** 3), total_bytes * 1.35 + 1 * (1024 ** 3)))


def write_manifest(save_dir, split_results):
    manifest = {
        "format": "split_frame_lmdb_v1",
        "splits": {
            split_name: {
                "path": result["path"],
                "count": result["count"],
            }
            for split_name, result in split_results.items()
        },
    }
    manifest_path = Path(save_dir) / FRAME_CACHE_MANIFEST
    with open(manifest_path, "w", encoding="utf-8") as outfile:
        json.dump(manifest, outfile, indent=2)
        outfile.write("\n")


def precache_split(config, split_name, save_dir, args):
    if lmdb is None:
        raise RuntimeError("lmdb is required for frame cache. Install it with `pip install lmdb`.")

    dataset = build_dataset(config, split_name, args)
    print(
        f"[FrameCache]: split={split_name} dataset_split={dataset.split} "
        f"samples={len(dataset)} save_dir={save_dir}"
    )

    db_name = f"{split_name}.lmdb"
    db_path = Path(save_dir) / db_name
    data_path = db_path / "data.mdb"
    if data_path.exists() and not args.overwrite:
        print(f"[FrameCache]: split={split_name} exists; pass --overwrite to rebuild.")
        return {"path": db_name, "count": len(dataset), "written": 0, "existing": len(dataset), "failed": 0}

    if db_path.exists() and args.overwrite:
        for name in ("data.mdb", "lock.mdb"):
            path = db_path / name
            if path.exists():
                os.remove(path)
    db_path.mkdir(parents=True, exist_ok=True)

    map_size = lmdb_map_size_bytes(dataset, override_gb=args.lmdb_map_size_gb)
    env = lmdb.open(
        str(db_path),
        subdir=True,
        map_size=map_size,
        meminit=False,
        map_async=True,
        writemap=False,
        max_dbs=1,
    )

    start = time.time()
    written = 0
    failed = 0
    failures = []
    commit_interval = max(int(args.lmdb_commit_interval), 1)
    txn = env.begin(write=True)
    try:
        for index, item in enumerate(dataset.lines, start=1):
            try:
                with open(item["frame_path"], "rb") as infile:
                    frame_bytes = infile.read()
                key = crop_cache_key(item["subject"], item["frame"]).encode("utf-8")
                txn.put(key, frame_bytes, overwrite=True)
                written += 1
            except Exception as exc:  # pragma: no cover - long-running data job
                failed += 1
                failures.append({"subject": item.get("subject"), "frame": item.get("frame"), "error": str(exc)})
                if args.fail_fast:
                    raise

            if index % commit_interval == 0:
                txn.commit()
                txn = env.begin(write=True)

            if args.log_interval > 0 and (index == 1 or index % args.log_interval == 0 or index == len(dataset)):
                elapsed = max(time.time() - start, 1.0e-6)
                rate = index / elapsed
                print(
                    f"[FrameCache]: split={split_name} {index}/{len(dataset)} "
                    f"written={written} failed={failed} rate={rate:.1f}/s"
                )
        txn.commit()
        txn = None
    finally:
        if txn is not None:
            txn.abort()
        env.sync()
        env.close()

    return {
        "path": db_name,
        "count": len(dataset),
        "written": written,
        "existing": 0,
        "failed": failed,
        "seconds": time.time() - start,
        "failures": failures[:100],
    }


def main():
    args = parse_args()
    config = load_config(args.config)
    save_dir = args.save_dir or config.get("data", {}).get("frame_cache_dir") or "outputs/mpcas_gaze/cache/frame_lmdb"
    splits = SPLIT_NAMES if args.split == "all" else (args.split,)

    os.makedirs(save_dir, exist_ok=True)
    results = {}
    for split_name in splits:
        results[split_name] = precache_split(config, split_name, save_dir, args)
    write_manifest(save_dir, results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
