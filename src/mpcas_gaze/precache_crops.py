import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import yaml

try:
    import lmdb
except ImportError:  # pragma: no cover
    lmdb = None

try:
    from .dataset import (
        CROP_CACHE_MANIFEST,
        CROP_CACHE_SIZE,
        CoGazeDataset,
        crop,
        crop_cache_key,
        crop_cache_path,
        pack_cached_crops,
        pack_jpeg_cached_crops,
        preprocess_crop_uint8,
        read_image_rgb,
        unpack_cached_crops,
        unpack_jpeg_cached_crops,
    )
except ImportError:  # Allow direct script execution.
    from dataset import (
        CROP_CACHE_MANIFEST,
        CROP_CACHE_SIZE,
        CoGazeDataset,
        crop,
        crop_cache_key,
        crop_cache_path,
        pack_cached_crops,
        pack_jpeg_cached_crops,
        preprocess_crop_uint8,
        read_image_rgb,
        unpack_cached_crops,
        unpack_jpeg_cached_crops,
    )


SPLIT_NAMES = ("train", "val", "test")


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-crop CoGaze face/eye images into cache files.")
    parser.add_argument("--config", default="configs/mpcas_gaze_example.yaml")
    parser.add_argument("--split", default="all", choices=["all", *SPLIT_NAMES])
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--format", default="files", choices=["files", "memmap", "lmdb", "lmdb_jpeg"])
    parser.add_argument("--source-cache-dir", default=None)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--lmdb-map-size-gb", type=float, default=None)
    parser.add_argument("--lmdb-commit-interval", type=int, default=512)
    parser.add_argument("--data-path", default=None)
    parser.add_argument(
        "--device-type",
        default=None,
        choices=["phone", "tablet", "unknown"],
        help="Filter cached samples by classified device type.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--log-interval", type=int, default=1000)
    return parser.parse_args()


class CropCacheSource:
    def __init__(self, cache_dir, split_name):
        self.cache_dir = str(cache_dir) if cache_dir else None
        self.split_name = split_name
        self.mode = None
        self.index = None
        self.array = None
        self.env = None
        if not self.cache_dir:
            return

        manifest_path = Path(self.cache_dir) / CROP_CACHE_MANIFEST
        if not manifest_path.exists():
            self.mode = "files"
            return

        manifest = load_config(manifest_path)
        if int(manifest.get("crop_cache_size", -1)) != CROP_CACHE_SIZE:
            raise ValueError(
                f"Source crop cache size mismatch: manifest={manifest.get('crop_cache_size')} expected={CROP_CACHE_SIZE}"
            )
        split_entry = manifest.get("splits", {}).get(split_name)
        if split_entry is None:
            raise KeyError(f"Source cache has no split {split_name!r}: {manifest_path}")

        if manifest.get("format") == "split_memmap_v1":
            self.mode = "memmap"
            self.index = load_config(Path(self.cache_dir) / split_entry["index"])
            self.array = np.load(Path(self.cache_dir) / split_entry["file"], mmap_mode="r", allow_pickle=False)
            return

        if manifest.get("format") in {"split_lmdb_v1", "split_lmdb_jpeg_v1"}:
            if lmdb is None:
                raise RuntimeError("lmdb is required to read source LMDB cache. Install it with `pip install lmdb`.")
            self.mode = "lmdb_jpeg" if manifest.get("format") == "split_lmdb_jpeg_v1" else "lmdb"
            self.env = lmdb.open(
                str(Path(self.cache_dir) / split_entry["path"]),
                subdir=True,
                readonly=True,
                lock=False,
                readahead=False,
                max_readers=2048,
            )
            return

        raise ValueError(f"Unsupported source cache format: {manifest.get('format')}")

    def get(self, item):
        if not self.cache_dir:
            return None
        key = crop_cache_key(item["subject"], item["frame"])
        if self.mode == "files":
            source_path = crop_cache_path(self.cache_dir, item["subject"], item["frame"])
            if not os.path.exists(source_path):
                raise FileNotFoundError(source_path)
            packed = np.load(source_path, allow_pickle=False)
        elif self.mode == "memmap":
            row = self.index.get(key)
            if row is None:
                raise FileNotFoundError(f"Missing source memmap row for {key}")
            packed = self.array[int(row)]
        elif self.mode == "lmdb":
            with self.env.begin(write=False) as txn:
                value = txn.get(key.encode("utf-8"))
            if value is None:
                raise FileNotFoundError(f"Missing source LMDB row for {key}")
            packed = np.frombuffer(value, dtype=np.uint8)
        elif self.mode == "lmdb_jpeg":
            with self.env.begin(write=False) as txn:
                value = txn.get(key.encode("utf-8"))
            if value is None:
                raise FileNotFoundError(f"Missing source JPEG LMDB row for {key}")
            face, left, right = unpack_jpeg_cached_crops(value)
            packed = pack_cached_crops(face, left, right)
        else:
            return None
        if packed.dtype != np.uint8 or packed.size != CROP_CACHE_SIZE:
            raise ValueError(f"Invalid source crop cache for {key}: dtype={packed.dtype}, size={packed.size}")
        return packed.reshape(-1)

    def get_jpeg(self, item, quality=90):
        if not self.cache_dir:
            return None
        key = crop_cache_key(item["subject"], item["frame"])
        if self.mode == "lmdb_jpeg":
            with self.env.begin(write=False) as txn:
                value = txn.get(key.encode("utf-8"))
            if value is None:
                raise FileNotFoundError(f"Missing source JPEG LMDB row for {key}")
            return value
        packed = self.get(item)
        if packed is None:
            return None
        face, left, right = unpack_cached_crops(packed)
        return pack_jpeg_cached_crops(face, left, right, quality=quality)


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
    )


def save_cached_sample(item, save_dir, overwrite=False):
    output_path = crop_cache_path(save_dir, item["subject"], item["frame"])
    if os.path.exists(output_path) and not overwrite:
        return "exists", output_path

    image = read_image_rgb(item["frame_path"])
    face = preprocess_crop_uint8(crop(image, item["boxes"][0]), (224, 224))
    left = preprocess_crop_uint8(crop(image, item["boxes"][1]), (112, 112))
    right = preprocess_crop_uint8(crop(image, item["boxes"][2]), (112, 112), flip=True)
    packed = pack_cached_crops(face, left, right)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp_path = f"{output_path}.tmp.npy"
    try:
        with open(tmp_path, "wb") as outfile:
            # np.save accepts file objects and avoids appending another suffix.
            np.save(outfile, packed, allow_pickle=False)
        os.replace(tmp_path, output_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return "written", output_path


def load_or_build_packed_sample(item, source=None):
    if source is not None:
        packed = source.get(item)
        if packed is not None:
            return packed

    image = read_image_rgb(item["frame_path"])
    face = preprocess_crop_uint8(crop(image, item["boxes"][0]), (224, 224))
    left = preprocess_crop_uint8(crop(image, item["boxes"][1]), (112, 112))
    right = preprocess_crop_uint8(crop(image, item["boxes"][2]), (112, 112), flip=True)
    return pack_cached_crops(face, left, right)


def load_or_build_jpeg_sample(item, source=None, quality=90):
    if source is not None:
        packed_jpeg = source.get_jpeg(item, quality=quality)
        if packed_jpeg is not None:
            return packed_jpeg

    image = read_image_rgb(item["frame_path"])
    face = preprocess_crop_uint8(crop(image, item["boxes"][0]), (224, 224))
    left = preprocess_crop_uint8(crop(image, item["boxes"][1]), (112, 112))
    right = preprocess_crop_uint8(crop(image, item["boxes"][2]), (112, 112), flip=True)
    return pack_jpeg_cached_crops(face, left, right, quality=quality)


def write_manifest(save_dir, split_name, dataset, result):
    manifest_path = Path(save_dir) / f"manifest_{split_name}.json"
    manifest = {
        "split": split_name,
        "dataset_split": dataset.split,
        "samples": len(dataset),
        **result,
    }
    with open(manifest_path, "w", encoding="utf-8") as outfile:
        json.dump(manifest, outfile, indent=2)
        outfile.write("\n")


def write_merged_manifest(save_dir, split_results):
    manifest = {
        "format": "split_memmap_v1",
        "crop_cache_size": CROP_CACHE_SIZE,
        "splits": {
            split_name: {
                "file": result["file"],
                "index": result["index"],
                "count": result["count"],
            }
            for split_name, result in split_results.items()
        },
    }
    manifest_path = Path(save_dir) / CROP_CACHE_MANIFEST
    with open(manifest_path, "w", encoding="utf-8") as outfile:
        json.dump(manifest, outfile, indent=2)
        outfile.write("\n")


def write_lmdb_manifest(save_dir, split_results):
    manifest = {
        "format": "split_lmdb_v1",
        "crop_cache_size": CROP_CACHE_SIZE,
        "splits": {
            split_name: {
                "path": result["path"],
                "count": result["count"],
            }
            for split_name, result in split_results.items()
        },
    }
    manifest_path = Path(save_dir) / CROP_CACHE_MANIFEST
    with open(manifest_path, "w", encoding="utf-8") as outfile:
        json.dump(manifest, outfile, indent=2)
        outfile.write("\n")


def write_lmdb_jpeg_manifest(save_dir, split_results, jpeg_quality):
    manifest = {
        "format": "split_lmdb_jpeg_v1",
        "crop_cache_size": CROP_CACHE_SIZE,
        "jpeg_quality": int(jpeg_quality),
        "splits": {
            split_name: {
                "path": result["path"],
                "count": result["count"],
            }
            for split_name, result in split_results.items()
        },
    }
    manifest_path = Path(save_dir) / CROP_CACHE_MANIFEST
    with open(manifest_path, "w", encoding="utf-8") as outfile:
        json.dump(manifest, outfile, indent=2)
        outfile.write("\n")


def precache_split(config, split_name, save_dir, args):
    dataset = build_dataset(config, split_name, args)
    print(f"[Cache]: split={split_name} dataset_split={dataset.split} samples={len(dataset)} save_dir={save_dir}")

    start = time.time()
    written = 0
    existing = 0
    failed = 0
    failures = []

    for index, item in enumerate(dataset.lines, start=1):
        try:
            status, _ = save_cached_sample(item, save_dir, overwrite=args.overwrite)
            if status == "written":
                written += 1
            else:
                existing += 1
        except Exception as exc:  # pragma: no cover - long-running data job
            failed += 1
            failures.append({"subject": item.get("subject"), "frame": item.get("frame"), "error": str(exc)})
            if args.fail_fast:
                raise

        if args.log_interval > 0 and (index == 1 or index % args.log_interval == 0 or index == len(dataset)):
            elapsed = max(time.time() - start, 1.0e-6)
            rate = index / elapsed
            print(
                f"[Cache]: split={split_name} {index}/{len(dataset)} "
                f"written={written} existing={existing} failed={failed} rate={rate:.1f}/s"
            )

    result = {
        "written": written,
        "existing": existing,
        "failed": failed,
        "seconds": time.time() - start,
        "failures": failures[:100],
    }
    write_manifest(save_dir, split_name, dataset, result)
    return result


def precache_split_memmap(config, split_name, save_dir, args):
    dataset = build_dataset(config, split_name, args)
    source = CropCacheSource(args.source_cache_dir, split_name) if args.source_cache_dir else None
    print(
        f"[CacheMerge]: split={split_name} dataset_split={dataset.split} "
        f"samples={len(dataset)} save_dir={save_dir}"
    )

    array_file = f"{split_name}_crops.npy"
    index_file = f"{split_name}_index.json"
    array_path = Path(save_dir) / array_file
    index_path = Path(save_dir) / index_file
    if array_path.exists() and index_path.exists() and not args.overwrite:
        print(f"[CacheMerge]: split={split_name} exists; pass --overwrite to rebuild.")
        return {"file": array_file, "index": index_file, "count": len(dataset), "written": 0, "existing": len(dataset), "failed": 0}

    tmp_array_path = Path(save_dir) / f"{split_name}_crops.tmp.npy"
    tmp_index_path = Path(save_dir) / f"{split_name}_index.tmp.json"
    for path in (tmp_array_path, tmp_index_path):
        if path.exists():
            os.remove(path)

    start = time.time()
    failed = 0
    failures = []
    index_map = {}
    crops = np.lib.format.open_memmap(
        str(tmp_array_path),
        mode="w+",
        dtype=np.uint8,
        shape=(len(dataset), CROP_CACHE_SIZE),
    )

    for row, item in enumerate(dataset.lines):
        try:
            crops[row, :] = load_or_build_packed_sample(item, source=source)
            index_map[crop_cache_key(item["subject"], item["frame"])] = row
        except Exception as exc:  # pragma: no cover - long-running data job
            failed += 1
            failures.append({"subject": item.get("subject"), "frame": item.get("frame"), "error": str(exc)})
            if args.fail_fast:
                raise

        index = row + 1
        if args.log_interval > 0 and (index == 1 or index % args.log_interval == 0 or index == len(dataset)):
            elapsed = max(time.time() - start, 1.0e-6)
            rate = index / elapsed
            print(
                f"[CacheMerge]: split={split_name} {index}/{len(dataset)} "
                f"failed={failed} rate={rate:.1f}/s"
            )

    crops.flush()
    del crops
    with open(tmp_index_path, "w", encoding="utf-8") as outfile:
        json.dump(index_map, outfile, separators=(",", ":"))
        outfile.write("\n")

    os.replace(tmp_array_path, array_path)
    os.replace(tmp_index_path, index_path)
    result = {
        "file": array_file,
        "index": index_file,
        "count": len(dataset),
        "written": len(dataset) - failed,
        "existing": 0,
        "failed": failed,
        "seconds": time.time() - start,
        "failures": failures[:100],
    }
    write_manifest(save_dir, split_name, dataset, result)
    return result


def lmdb_map_size_bytes(num_samples, override_gb=None):
    if override_gb is not None:
        return int(float(override_gb) * (1024 ** 3))
    payload = num_samples * CROP_CACHE_SIZE
    # LMDB map_size is an upper bound. On 64-bit Linux it does not allocate
    # the full space immediately, so keep comfortable headroom for B+tree pages.
    return int(max(1 * (1024 ** 3), payload * 1.25 + 1 * (1024 ** 3)))


def lmdb_jpeg_map_size_bytes(num_samples, override_gb=None):
    if override_gb is not None:
        return int(float(override_gb) * (1024 ** 3))
    # JPEG crops are much smaller than raw uint8 crops, which is the point of
    # this HDD-friendly cache. Keep enough headroom for high-texture frames.
    payload = num_samples * 96 * 1024
    return int(max(1 * (1024 ** 3), payload * 1.5 + 1 * (1024 ** 3)))


def precache_split_lmdb(config, split_name, save_dir, args):
    if lmdb is None:
        raise RuntimeError("lmdb is required for `--format lmdb`. Install it with `pip install lmdb`.")

    dataset = build_dataset(config, split_name, args)
    source = CropCacheSource(args.source_cache_dir, split_name) if args.source_cache_dir else None
    print(
        f"[CacheLMDB]: split={split_name} dataset_split={dataset.split} "
        f"samples={len(dataset)} save_dir={save_dir}"
    )

    db_name = f"{split_name}.lmdb"
    db_path = Path(save_dir) / db_name
    data_path = db_path / "data.mdb"
    if data_path.exists() and not args.overwrite:
        print(f"[CacheLMDB]: split={split_name} exists; pass --overwrite to rebuild.")
        return {"path": db_name, "count": len(dataset), "written": 0, "existing": len(dataset), "failed": 0}

    if db_path.exists() and args.overwrite:
        # Remove only the files LMDB creates inside this split-specific directory.
        for name in ("data.mdb", "lock.mdb"):
            path = db_path / name
            if path.exists():
                os.remove(path)
    db_path.mkdir(parents=True, exist_ok=True)

    map_size = lmdb_map_size_bytes(len(dataset), override_gb=args.lmdb_map_size_gb)
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
                packed = load_or_build_packed_sample(item, source=source)
                key = crop_cache_key(item["subject"], item["frame"]).encode("utf-8")
                txn.put(key, packed.tobytes(order="C"), overwrite=True)
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
                    f"[CacheLMDB]: split={split_name} {index}/{len(dataset)} "
                    f"written={written} failed={failed} rate={rate:.1f}/s"
                )
        txn.commit()
        txn = None
    finally:
        if txn is not None:
            txn.abort()
        env.sync()
        env.close()

    result = {
        "path": db_name,
        "count": len(dataset),
        "written": written,
        "existing": 0,
        "failed": failed,
        "seconds": time.time() - start,
        "failures": failures[:100],
    }
    write_manifest(save_dir, split_name, dataset, result)
    return result


def precache_split_lmdb_jpeg(config, split_name, save_dir, args):
    if lmdb is None:
        raise RuntimeError("lmdb is required for `--format lmdb_jpeg`. Install it with `pip install lmdb`.")

    dataset = build_dataset(config, split_name, args)
    source = CropCacheSource(args.source_cache_dir, split_name) if args.source_cache_dir else None
    print(
        f"[CacheJPEG-LMDB]: split={split_name} dataset_split={dataset.split} "
        f"samples={len(dataset)} save_dir={save_dir} quality={int(args.jpeg_quality)}"
    )

    db_name = f"{split_name}.lmdb"
    db_path = Path(save_dir) / db_name
    data_path = db_path / "data.mdb"
    if data_path.exists() and not args.overwrite:
        print(f"[CacheJPEG-LMDB]: split={split_name} exists; pass --overwrite to rebuild.")
        return {"path": db_name, "count": len(dataset), "written": 0, "existing": len(dataset), "failed": 0}

    if db_path.exists() and args.overwrite:
        for name in ("data.mdb", "lock.mdb"):
            path = db_path / name
            if path.exists():
                os.remove(path)
    db_path.mkdir(parents=True, exist_ok=True)

    map_size = lmdb_jpeg_map_size_bytes(len(dataset), override_gb=args.lmdb_map_size_gb)
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
                packed = load_or_build_jpeg_sample(item, source=source, quality=args.jpeg_quality)
                key = crop_cache_key(item["subject"], item["frame"]).encode("utf-8")
                txn.put(key, packed, overwrite=True)
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
                    f"[CacheJPEG-LMDB]: split={split_name} {index}/{len(dataset)} "
                    f"written={written} failed={failed} rate={rate:.1f}/s"
                )
        txn.commit()
        txn = None
    finally:
        if txn is not None:
            txn.abort()
        env.sync()
        env.close()

    result = {
        "path": db_name,
        "count": len(dataset),
        "written": written,
        "existing": 0,
        "failed": failed,
        "seconds": time.time() - start,
        "failures": failures[:100],
    }
    write_manifest(save_dir, split_name, dataset, result)
    return result


def main():
    args = parse_args()
    config = load_config(args.config)
    save_dir = args.save_dir or config.get("data", {}).get("crop_cache_dir") or "outputs/mpcas_gaze/cache/crops_npy"
    splits = SPLIT_NAMES if args.split == "all" else (args.split,)

    os.makedirs(save_dir, exist_ok=True)
    results = {}
    for split_name in splits:
        if args.format == "memmap":
            results[split_name] = precache_split_memmap(config, split_name, save_dir, args)
        elif args.format == "lmdb":
            results[split_name] = precache_split_lmdb(config, split_name, save_dir, args)
        elif args.format == "lmdb_jpeg":
            results[split_name] = precache_split_lmdb_jpeg(config, split_name, save_dir, args)
        else:
            results[split_name] = precache_split(config, split_name, save_dir, args)
    if args.format == "memmap":
        write_merged_manifest(save_dir, results)
    elif args.format == "lmdb":
        write_lmdb_manifest(save_dir, results)
    elif args.format == "lmdb_jpeg":
        write_lmdb_jpeg_manifest(save_dir, results, jpeg_quality=args.jpeg_quality)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
