import copy
import csv
import io
import json
import os
import random
import struct
import tarfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    import lmdb
except ImportError:  # pragma: no cover
    lmdb = None


DEFAULT_BAD_SUBJECTS = {"01185", "01730", "02065", "02585"}
FACE_CACHE_SHAPE = (3, 224, 224)
EYE_CACHE_SHAPE = (3, 112, 112)
FACE_CACHE_SIZE = int(np.prod(FACE_CACHE_SHAPE))
EYE_CACHE_SIZE = int(np.prod(EYE_CACHE_SHAPE))
CROP_CACHE_SIZE = FACE_CACHE_SIZE + 2 * EYE_CACHE_SIZE
CROP_CACHE_MANIFEST = "crop_cache_manifest.json"
FRAME_CACHE_MANIFEST = "frame_cache_manifest.json"
DEVICE_TYPE_ALIASES = {
    "phone": "phone",
    "mobile": "phone",
    "smartphone": "phone",
    "iphone": "phone",
    "ipod": "phone",
    "tablet": "tablet",
    "pad": "tablet",
    "ipad": "tablet",
    "unknown": "unknown",
}


def classify_device(device_name):
    name = (device_name or "").strip().lower()
    if name in {"sm-s9380"}:
        return "phone"
    if name in {"sm-x926c"}:
        return "tablet"
    if name.startswith("ipad") or "tablet" in name:
        return "tablet"
    if name.startswith("iphone") or name.startswith("ipod") or name.startswith("sm-") or "phone" in name:
        return "phone"
    return "unknown"


def normalize_device_type_filter(device_type_filter):
    if device_type_filter is None:
        return None
    if isinstance(device_type_filter, str):
        raw_values = device_type_filter.replace(";", ",").split(",")
    else:
        raw_values = device_type_filter

    normalized = set()
    for value in raw_values:
        if value is None:
            continue
        key = str(value).strip().lower()
        if not key or key in {"all", "any", "none", "null"}:
            continue
        normalized.add(DEVICE_TYPE_ALIASES.get(key, key))

    valid = {"phone", "tablet", "unknown"}
    invalid = sorted(normalized - valid)
    if invalid:
        raise ValueError(
            f"Unsupported device_type_filter values: {invalid}. "
            "Use phone, tablet, unknown, or a comma-separated combination."
        )
    return normalized or None


def read_json(path):
    with open(path, "r", encoding="utf-8") as infile:
        return json.load(infile)


def decode_image_rgb(data):
    if cv2 is not None:
        encoded = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image bytes")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with Image.open(io.BytesIO(data)) as image:
        return np.asarray(image.convert("RGB"))


def read_image_rgb(path):
    if cv2 is not None:
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def resize_rgb(image, size):
    if cv2 is not None:
        return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    return np.asarray(Image.fromarray(image).resize(size, Image.BILINEAR))


def crop(image, box):
    height, width = image.shape[:2]
    top = max(0, int(round(box[0])))
    bottom = min(height, int(round(box[1])))
    left = max(0, int(round(box[2])))
    right = min(width, int(round(box[3])))
    if bottom <= top or right <= left:
        raise ValueError(f"Invalid crop box {box} for image size {width}x{height}")
    return image[top:bottom, left:right]


def preprocess_crop(image, size, flip=False, brightness=0.0, contrast=0.0):
    image = resize_rgb(image, size)
    if flip:
        image = np.ascontiguousarray(image[:, ::-1, :])
    image = image.astype(np.float32, copy=False)
    image *= 1.0 / 255.0
    if brightness:
        image += random.uniform(-brightness, brightness)
    if contrast:
        scale = 1.0 + random.uniform(-contrast, contrast)
        image -= 0.5
        image *= scale
        image += 0.5
    np.clip(image, 0.0, 1.0, out=image)
    return image.transpose(2, 0, 1)


def preprocess_crop_uint8(image, size, flip=False):
    image = resize_rgb(image, size)
    if flip:
        image = np.ascontiguousarray(image[:, ::-1, :])
    return np.ascontiguousarray(image.transpose(2, 0, 1))


def crop_cache_path(cache_dir, subject, frame):
    frame_stem = os.path.splitext(str(frame))[0]
    return os.path.join(str(cache_dir), str(subject).zfill(5), f"{frame_stem}.npy")


def crop_cache_key(subject, frame):
    return f"{str(subject).zfill(5)}/{str(frame)}"


def pack_cached_crops(face, left, right):
    face = np.asarray(face, dtype=np.uint8).reshape(-1)
    left = np.asarray(left, dtype=np.uint8).reshape(-1)
    right = np.asarray(right, dtype=np.uint8).reshape(-1)
    return np.concatenate([face, left, right])


def _chw_to_hwc_uint8(image):
    image = np.asarray(image, dtype=np.uint8)
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected CHW RGB uint8 crop, got shape={image.shape}")
    return np.ascontiguousarray(image.transpose(1, 2, 0))


def _encode_jpeg_rgb(image, quality=90):
    image = np.asarray(image, dtype=np.uint8)
    quality = int(quality)
    if cv2 is not None:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise ValueError("Failed to encode crop as JPEG")
        return encoded.tobytes()

    with io.BytesIO() as buffer:
        Image.fromarray(image, mode="RGB").save(buffer, format="JPEG", quality=quality)
        return buffer.getvalue()


def _decode_jpeg_rgb_chw(data):
    if cv2 is not None:
        encoded = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode cached JPEG crop")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        with Image.open(io.BytesIO(data)) as image_file:
            image = np.asarray(image_file.convert("RGB"))
    return np.ascontiguousarray(image.transpose(2, 0, 1))


def pack_jpeg_cached_crops(face, left, right, quality=90):
    face_jpeg = _encode_jpeg_rgb(_chw_to_hwc_uint8(face), quality=quality)
    left_jpeg = _encode_jpeg_rgb(_chw_to_hwc_uint8(left), quality=quality)
    right_jpeg = _encode_jpeg_rgb(_chw_to_hwc_uint8(right), quality=quality)
    header = struct.pack("<III", len(face_jpeg), len(left_jpeg), len(right_jpeg))
    return header + face_jpeg + left_jpeg + right_jpeg


def unpack_jpeg_cached_crops(packed):
    if len(packed) < 12:
        raise ValueError(f"Invalid JPEG crop cache row: size={len(packed)}")
    face_size, left_size, right_size = struct.unpack("<III", packed[:12])
    offset = 12
    face_end = offset + face_size
    left_end = face_end + left_size
    right_end = left_end + right_size
    if right_end != len(packed):
        raise ValueError(
            f"Invalid JPEG crop cache row: header sizes total {right_end}, payload size {len(packed)}"
        )
    face = _decode_jpeg_rgb_chw(packed[offset:face_end])
    left = _decode_jpeg_rgb_chw(packed[face_end:left_end])
    right = _decode_jpeg_rgb_chw(packed[left_end:right_end])
    return face, left, right


def unpack_cached_crops(packed):
    packed = np.asarray(packed)
    if packed.dtype != np.uint8 or packed.size != CROP_CACHE_SIZE:
        raise ValueError(
            f"Invalid crop cache array: dtype={packed.dtype} size={packed.size}, "
            f"expected uint8 size={CROP_CACHE_SIZE}"
        )
    face_end = FACE_CACHE_SIZE
    left_end = face_end + EYE_CACHE_SIZE
    face = packed[:face_end].reshape(FACE_CACHE_SHAPE)
    left = packed[face_end:left_end].reshape(EYE_CACHE_SHAPE)
    right = packed[left_end:].reshape(EYE_CACHE_SHAPE)
    return face, left, right


def cached_crop_to_float(image, brightness=0.0, contrast=0.0):
    image = np.asarray(image, dtype=np.float32) * (1.0 / 255.0)
    if brightness:
        image += random.uniform(-brightness, brightness)
    if contrast:
        scale = 1.0 + random.uniform(-contrast, contrast)
        image -= 0.5
        image *= scale
        image += 0.5
    np.clip(image, 0.0, 1.0, out=image)
    return image


def cached_crop_to_uint8(image):
    return np.array(image, dtype=np.uint8, copy=True)


def image_batch_to_float(image, device, non_blocking=True, augment=False, brightness=0.0, contrast=0.0):
    image = image.to(device, non_blocking=non_blocking)
    can_augment = image.dtype == torch.uint8
    if image.dtype == torch.uint8:
        image = image.float()
        image.mul_(1.0 / 255.0)
    elif not image.dtype.is_floating_point:
        image = image.float()

    if can_augment and augment and brightness:
        delta = torch.empty((image.size(0), 1, 1, 1), device=image.device, dtype=image.dtype)
        delta.uniform_(-float(brightness), float(brightness))
        image.add_(delta)
    if can_augment and augment and contrast:
        scale = torch.empty((image.size(0), 1, 1, 1), device=image.device, dtype=image.dtype)
        scale.uniform_(1.0 - float(contrast), 1.0 + float(contrast))
        image.sub_(0.5).mul_(scale).add_(0.5)
    if can_augment and augment and (brightness or contrast):
        image.clamp_(0.0, 1.0)
    return image


def default_label_path():
    path = Path(__file__).resolve().parent.parent / "AFF-Net" / "label" / "Labels.tar.gz"
    return str(path) if path.exists() else None


def resolve_label_path(label_path):
    default_path = default_label_path()
    if isinstance(label_path, str) and label_path.strip().lower() in {"none", "null", "false", "off"}:
        return None
    if not label_path:
        return default_path

    raw_path = Path(str(label_path))
    candidates = [raw_path] if raw_path.is_absolute() else [
        Path.cwd() / raw_path,
        Path(__file__).resolve().parent.parent / raw_path,
        Path(__file__).resolve().parent / raw_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    if default_path:
        print(f"[Read Data]: label_path not found: {label_path}; using {default_path}")
        return default_path

    print(f"[Read Data]: label_path not found: {label_path}; falling back to raw apple annotations if available")
    return str(raw_path)


def load_newfaceldmk(subject_path, subject, label_path, tar_obj=None, tar_names=None):
    local_path = os.path.join(subject_path, "newFaceLdmk.json")
    if os.path.exists(local_path):
        return read_json(local_path)
    if not label_path:
        raise FileNotFoundError(local_path)
    label_path = os.path.abspath(label_path)
    if os.path.isdir(label_path):
        label_file = os.path.join(label_path, subject, "newFaceLdmk.json")
        if os.path.exists(label_file):
            return read_json(label_file)
        raise FileNotFoundError(label_file)
    member_name = f"{subject}/newFaceLdmk.json"
    if tar_obj is not None and member_name in tar_names:
        extracted = tar_obj.extractfile(member_name)
        if extracted is not None:
            return json.load(extracted)
    raise FileNotFoundError(member_name)


def normalize_subject_list(subject_list):
    if subject_list is None:
        return None
    if isinstance(subject_list, (str, os.PathLike)) and os.path.exists(subject_list):
        with open(subject_list, "r", encoding="utf-8") as infile:
            subjects = [line.strip() for line in infile if line.strip()]
    elif isinstance(subject_list, str):
        subjects = [item.strip() for item in subject_list.split(",") if item.strip()]
    else:
        subjects = list(subject_list)
    return {str(subject).zfill(5) for subject in subjects}


def load_aux_features(aux_path):
    if not aux_path:
        return {}, 0
    aux_path = str(aux_path)
    suffix = Path(aux_path).suffix.lower()
    table = {}
    aux_dim = 0
    if suffix == ".csv":
        with open(aux_path, "r", newline="", encoding="utf-8") as infile:
            reader = csv.DictReader(infile)
            aux_columns = [name for name in reader.fieldnames or [] if name.startswith("aux")]
            if not aux_columns:
                aux_columns = [name for name in reader.fieldnames or [] if name not in {"subject", "frame", "frameIndex"}]
            for row in reader:
                subject = str(row.get("subject", "")).zfill(5)
                frame = str(row.get("frame", row.get("frameIndex", "")))
                if frame.isdigit():
                    frame = f"{int(frame):05d}.jpg"
                values = np.array([float(row[name]) for name in aux_columns], dtype=np.float32)
                table[(subject, frame)] = values
                aux_dim = len(values)
    elif suffix == ".json":
        data = read_json(aux_path)
        for key, values in data.items():
            if isinstance(key, str) and "/" in key:
                subject, frame = key.split("/", 1)
            else:
                subject, frame = key
            values = np.asarray(values, dtype=np.float32)
            table[(str(subject).zfill(5), str(frame))] = values
            aux_dim = len(values)
    elif suffix == ".npz":
        data = np.load(aux_path, allow_pickle=True)
        if {"subjects", "frames", "features"}.issubset(set(data.files)):
            subjects = data["subjects"]
            frames = data["frames"]
            features = data["features"].astype(np.float32)
            for subject, frame, values in zip(subjects, frames, features):
                frame = str(frame)
                if frame.isdigit():
                    frame = f"{int(frame):05d}.jpg"
                table[(str(subject).zfill(5), frame)] = values
            aux_dim = int(features.shape[1])
        else:
            raise ValueError("NPZ aux file must contain subjects, frames and features arrays.")
    else:
        raise ValueError(f"Unsupported aux feature file: {aux_path}")
    return table, aux_dim


def jitter_boxes(boxes, rects, width, height, rects_normalized=True):
    boxes = copy.deepcopy(boxes)
    rects = np.asarray(rects, dtype=np.float32).copy()
    bbox = np.array(boxes, dtype=np.float32)
    bias = round(30 * random.uniform(-1, 1))
    bias = max(np.max(-bbox[0, [0, 2]]), bias)
    bias = max(np.max(-2 * bbox[1:, [0, 2]] + 0.5), bias)

    for index, scale in [(0, 1.0), (1, 0.5), (2, 0.5)]:
        delta = int(round(scale * bias))
        top, bottom, left, right = [int(round(value + delta)) for value in boxes[index]]
        box_h = max(bottom - top, 1)
        box_w = max(right - left, 1)

        if box_h >= height:
            top, bottom = 0, height
        else:
            if top < 0:
                bottom -= top
                top = 0
            if bottom > height:
                top -= bottom - height
                bottom = height
            top = max(0, top)
            bottom = min(height, bottom)
            if bottom <= top:
                bottom = min(height, top + 1)
                top = max(0, bottom - 1)

        if box_w >= width:
            left, right = 0, width
        else:
            if left < 0:
                right -= left
                left = 0
            if right > width:
                left -= right - width
                right = width
            left = max(0, left)
            right = min(width, right)
            if right <= left:
                right = min(width, left + 1)
                left = max(0, right - 1)

        boxes[index] = [top, bottom, left, right]

    x_denom = float(width) if rects_normalized else 1.0
    y_denom = float(height) if rects_normalized else 1.0
    for index, (top, bottom, left, right) in enumerate(boxes):
        offset = index * 4
        rects[offset] = (right - left) / x_denom
        rects[offset + 1] = (bottom - top) / y_denom
        rects[offset + 2] = left / x_denom
        rects[offset + 3] = top / y_denom
    return boxes, rects


class CoGazeDataset(Dataset):
    def __init__(
        self,
        data_paths,
        split,
        annotation_mode="auto",
        label_path=None,
        subject_list=None,
        device_filter=None,
        device_type_filter=None,
        orientation_filter=None,
        max_samples=None,
        iphone_only=False,
        aux_path=None,
        aux_dim=0,
        augment=True,
        brightness=0.0,
        contrast=0.0,
        crop_cache_dir=None,
        crop_cache_required=True,
        crop_cache_decode_on_gpu=True,
        frame_cache_dir=None,
        frame_cache_required=True,
        loader_threads=1,
    ):
        if isinstance(data_paths, (str, os.PathLike)):
            data_paths = [data_paths]
        self.data_paths = [os.path.abspath(str(path)) for path in data_paths]
        self.split = split
        self.annotation_mode = annotation_mode
        self.label_path = resolve_label_path(label_path)
        self.subject_filter = normalize_subject_list(subject_list)
        self.device_filter = device_filter
        self.device_type_filter = normalize_device_type_filter(device_type_filter)
        self.orientation_filter = set(orientation_filter or [])
        self.max_samples = max_samples
        self.iphone_only = iphone_only
        self.augment = augment and split != "test"
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.crop_cache_dir = str(crop_cache_dir) if crop_cache_dir else None
        self.crop_cache_required = bool(crop_cache_required)
        self.crop_cache_decode_on_gpu = bool(crop_cache_decode_on_gpu)
        self.crop_cache_mode = None
        self.crop_cache_index = None
        self.crop_cache_array_path = None
        self.crop_cache_array = None
        self.crop_cache_lmdb_path = None
        self.crop_cache_lmdb_env = None
        self.frame_cache_dir = str(frame_cache_dir) if frame_cache_dir else None
        self.frame_cache_required = bool(frame_cache_required)
        self.frame_cache_lmdb_path = None
        self.frame_cache_lmdb_env = None
        self.loader_threads = max(1, int(loader_threads or 1))
        self._loader_executor = None
        self.aux_table, detected_aux_dim = load_aux_features(aux_path)
        self.aux_dim = int(aux_dim or detected_aux_dim or 0)

        self.lines = []
        self.labels = {}
        self.screens = {}
        self.device_names = {}
        self.device_types = {}
        self.annotation_counts = {"newfaceldmk": 0, "apple": 0}
        self._build_index()
        self.device_type_counts = Counter(item["device_type"] for item in self.lines)
        self._init_crop_cache()
        self._init_frame_cache()
        if not self.lines:
            filters = []
            if self.device_type_filter:
                filters.append(f"device_type_filter={sorted(self.device_type_filter)}")
            if self.device_filter:
                filters.append(f"device_filter={self.device_filter!r}")
            filter_text = f" ({', '.join(filters)})" if filters else ""
            raise RuntimeError(f"No usable {split} samples found in {self.data_paths}{filter_text}")

    def _init_crop_cache(self):
        if not self.crop_cache_dir:
            return
        manifest_path = os.path.join(self.crop_cache_dir, CROP_CACHE_MANIFEST)
        if not os.path.exists(manifest_path):
            self.crop_cache_mode = "files"
            return

        manifest = read_json(manifest_path)
        if int(manifest.get("crop_cache_size", -1)) != CROP_CACHE_SIZE:
            raise ValueError(
                f"Crop cache size mismatch: manifest={manifest.get('crop_cache_size')} expected={CROP_CACHE_SIZE}"
            )

        manifest_format = manifest.get("format")
        split_entry = manifest.get("splits", {}).get(self.split)
        if split_entry is None:
            if self.crop_cache_required:
                raise KeyError(f"Crop cache manifest has no split entry for {self.split!r}: {manifest_path}")
            self.crop_cache_mode = None
            return

        if manifest_format == "split_memmap_v1":
            index_path = os.path.join(self.crop_cache_dir, split_entry["index"])
            array_path = os.path.join(self.crop_cache_dir, split_entry["file"])
            if self.crop_cache_required and (not os.path.exists(index_path) or not os.path.exists(array_path)):
                raise FileNotFoundError(f"Missing merged crop cache files for split {self.split}: {index_path}, {array_path}")
            self.crop_cache_index = read_json(index_path)
            self.crop_cache_array_path = array_path
            self.crop_cache_mode = "memmap"
            return

        if manifest_format in {"split_lmdb_v1", "split_lmdb_jpeg_v1"}:
            if lmdb is None:
                raise RuntimeError("lmdb is required for crop_cache mode 'lmdb'. Install it with `pip install lmdb`.")
            lmdb_path = os.path.join(self.crop_cache_dir, split_entry["path"])
            data_path = os.path.join(lmdb_path, "data.mdb")
            if self.crop_cache_required and not os.path.exists(data_path):
                raise FileNotFoundError(f"Missing LMDB crop cache for split {self.split}: {lmdb_path}")
            self.crop_cache_lmdb_path = lmdb_path
            self.crop_cache_mode = "lmdb_jpeg" if manifest_format == "split_lmdb_jpeg_v1" else "lmdb"
            return

        raise ValueError(f"Unsupported crop cache manifest format: {manifest_format}")

    def _init_frame_cache(self):
        if not self.frame_cache_dir:
            return
        if lmdb is None:
            raise RuntimeError("lmdb is required for frame_cache_dir. Install it with `pip install lmdb`.")
        manifest_path = os.path.join(self.frame_cache_dir, FRAME_CACHE_MANIFEST)
        if not os.path.exists(manifest_path):
            if self.frame_cache_required:
                raise FileNotFoundError(f"Missing frame cache manifest: {manifest_path}")
            return

        manifest = read_json(manifest_path)
        manifest_format = manifest.get("format")
        if manifest_format != "split_frame_lmdb_v1":
            raise ValueError(f"Unsupported frame cache manifest format: {manifest_format}")
        split_entry = manifest.get("splits", {}).get(self.split)
        if split_entry is None:
            if self.frame_cache_required:
                raise KeyError(f"Frame cache manifest has no split entry for {self.split!r}: {manifest_path}")
            return
        lmdb_path = os.path.join(self.frame_cache_dir, split_entry["path"])
        data_path = os.path.join(lmdb_path, "data.mdb")
        if self.frame_cache_required and not os.path.exists(data_path):
            raise FileNotFoundError(f"Missing LMDB frame cache for split {self.split}: {lmdb_path}")
        self.frame_cache_lmdb_path = lmdb_path

    def __getstate__(self):
        state = self.__dict__.copy()
        state["crop_cache_array"] = None
        state["crop_cache_lmdb_env"] = None
        state["frame_cache_lmdb_env"] = None
        state["_loader_executor"] = None
        return state

    def _get_crop_cache_array(self):
        if self.crop_cache_array is None:
            self.crop_cache_array = np.load(self.crop_cache_array_path, mmap_mode="r", allow_pickle=False)
            if self.crop_cache_array.dtype != np.uint8 or self.crop_cache_array.shape[1] != CROP_CACHE_SIZE:
                raise ValueError(
                    f"Invalid merged crop cache array {self.crop_cache_array_path}: "
                    f"dtype={self.crop_cache_array.dtype}, shape={self.crop_cache_array.shape}"
                )
        return self.crop_cache_array

    def _get_crop_cache_lmdb_env(self):
        if self.crop_cache_lmdb_env is None:
            self.crop_cache_lmdb_env = lmdb.open(
                self.crop_cache_lmdb_path,
                subdir=True,
                readonly=True,
                lock=False,
                readahead=False,
                max_readers=2048,
            )
        return self.crop_cache_lmdb_env

    def _get_frame_cache_lmdb_env(self):
        if self.frame_cache_lmdb_env is None:
            self.frame_cache_lmdb_env = lmdb.open(
                self.frame_cache_lmdb_path,
                subdir=True,
                readonly=True,
                lock=False,
                readahead=False,
                max_readers=2048,
            )
        return self.frame_cache_lmdb_env

    def _load_cached_crops(self, item):
        if not self.crop_cache_dir or not item["rects_normalized"]:
            return None
        if self.crop_cache_mode == "memmap":
            row = self.crop_cache_index.get(crop_cache_key(item["subject"], item["frame"]))
            if row is None:
                if self.crop_cache_required:
                    raise FileNotFoundError(f"Missing merged crop cache row for {item['subject']}/{item['frame']}")
                return None
            return self._get_crop_cache_array()[int(row)]
        if self.crop_cache_mode in {"lmdb", "lmdb_jpeg"}:
            key = crop_cache_key(item["subject"], item["frame"]).encode("utf-8")
            with self._get_crop_cache_lmdb_env().begin(write=False) as txn:
                value = txn.get(key)
            if value is None:
                if self.crop_cache_required:
                    raise FileNotFoundError(f"Missing LMDB crop cache row for {item['subject']}/{item['frame']}")
                return None
            if self.crop_cache_mode == "lmdb_jpeg":
                return value
            packed = np.frombuffer(value, dtype=np.uint8)
            if packed.size != CROP_CACHE_SIZE:
                raise ValueError(
                    f"Invalid LMDB crop cache row for {item['subject']}/{item['frame']}: "
                    f"size={packed.size}, expected={CROP_CACHE_SIZE}"
                )
            return packed

        path = crop_cache_path(self.crop_cache_dir, item["subject"], item["frame"])
        if os.path.exists(path):
            return np.load(path, allow_pickle=False)
        if self.crop_cache_required:
            raise FileNotFoundError(
                f"Missing crop cache for {item['subject']}/{item['frame']}: {path}. "
                "Run `python -m mpcas_gaze.precache_crops` first or disable data.crop_cache_dir."
            )
        return None

    def _load_frame_image(self, item):
        if self.frame_cache_lmdb_path:
            key = crop_cache_key(item["subject"], item["frame"]).encode("utf-8")
            with self._get_frame_cache_lmdb_env().begin(write=False) as txn:
                value = txn.get(key)
            if value is not None:
                return decode_image_rgb(value)
            if self.frame_cache_required:
                raise FileNotFoundError(f"Missing frame cache row for {item['subject']}/{item['frame']}")
        return read_image_rgb(item["frame_path"])

    def _subject_iter(self, data_path):
        subjects = sorted(os.listdir(data_path))
        for subject in subjects:
            subject = str(subject).zfill(5)
            if self.subject_filter and subject not in self.subject_filter:
                continue
            subject_path = os.path.join(data_path, subject)
            if not os.path.isdir(subject_path) or subject in DEFAULT_BAD_SUBJECTS:
                continue
            info_path = os.path.join(subject_path, "info.json")
            if not os.path.exists(info_path):
                continue
            info = read_json(info_path)
            if info.get("Dataset") != self.split:
                continue
            device_name = info.get("DeviceName", "")
            device_type = classify_device(device_name)
            if self.iphone_only and device_type != "phone":
                continue
            if self.device_type_filter and device_type not in self.device_type_filter:
                continue
            if self.device_filter and self.device_filter.lower() not in device_name.lower():
                continue
            yield subject, subject_path, info, device_name, device_type

    def _build_index(self):
        for data_path in self.data_paths:
            mode = self.annotation_mode
            if mode == "auto":
                start_count = len(self.lines)
                if self._has_newfaceldmk(data_path):
                    self._build_newfaceldmk_index(data_path)
                    self.annotation_counts["newfaceldmk"] += len(self.lines) - start_count
                if len(self.lines) == start_count:
                    apple_start_count = len(self.lines)
                    self._build_apple_index(data_path)
                    self.annotation_counts["apple"] += len(self.lines) - apple_start_count
                if self.max_samples is not None and len(self.lines) >= self.max_samples:
                    self.lines = self.lines[: self.max_samples]
                    break
                continue
            if mode == "newfaceldmk":
                start_count = len(self.lines)
                self._build_newfaceldmk_index(data_path)
                self.annotation_counts["newfaceldmk"] += len(self.lines) - start_count
            elif mode == "apple":
                start_count = len(self.lines)
                self._build_apple_index(data_path)
                self.annotation_counts["apple"] += len(self.lines) - start_count
            else:
                raise ValueError(f"Unsupported annotation mode: {self.annotation_mode}")
            if self.max_samples is not None and len(self.lines) >= self.max_samples:
                self.lines = self.lines[: self.max_samples]
                break

    def _has_newfaceldmk(self, data_path):
        if self.label_path and os.path.exists(self.label_path):
            return True
        for subject in os.listdir(data_path):
            if os.path.exists(os.path.join(data_path, subject, "newFaceLdmk.json")):
                return True
        return False

    def _build_newfaceldmk_index(self, data_path):
        tar_obj = None
        tar_names = set()
        if self.label_path and os.path.isfile(self.label_path):
            tar_obj = tarfile.open(self.label_path, "r:gz")
            tar_names = set(tar_obj.getnames())
        try:
            for subject, subject_path, info, device_name, device_type in self._subject_iter(data_path):
                try:
                    subject_lines = load_newfaceldmk(subject_path, subject, self.label_path, tar_obj, tar_names)
                except FileNotFoundError:
                    continue
                labels = read_json(os.path.join(subject_path, "dotInfo.json"))
                screen_path = os.path.join(subject_path, "screen.json")
                screen = read_json(screen_path) if os.path.exists(screen_path) else {}
                self.labels[subject] = labels
                self.screens[subject] = screen
                self.device_names[subject] = device_name
                self.device_types[subject] = device_type
                label_count = len(labels.get("XCam", []))
                orientations = screen.get("Orientation", [])
                dot_nums = labels.get("DotNum", [])
                tasks = labels.get("Task", [])
                phases = labels.get("Phase", [])
                att_valids = labels.get("AttValid", [])
                for line in subject_lines:
                    if len(line) < 7:
                        continue
                    frame = line[1]
                    frame_index = int(os.path.splitext(frame)[0])
                    if frame_index >= label_count:
                        continue
                    orientation = int(orientations[frame_index]) if frame_index < len(orientations) else -1
                    if self.orientation_filter and orientation not in self.orientation_filter:
                        continue
                    dot_num = int(dot_nums[frame_index]) if frame_index < len(dot_nums) else -1
                    task = str(tasks[frame_index]) if frame_index < len(tasks) else ""
                    phase = str(phases[frame_index]) if frame_index < len(phases) else ""
                    att_valid = int(att_valids[frame_index]) if frame_index < len(att_valids) else -1
                    self.lines.append(
                        {
                            "subject": subject,
                            "frame": frame,
                            "frame_index": frame_index,
                            "frame_path": os.path.join(subject_path, "frames", frame),
                            "boxes": [line[2], line[3], line[4]],
                            "rects": np.asarray(line[5], dtype=np.float32),
                            "rects_normalized": True,
                            "label": np.array([labels["XCam"][frame_index], labels["YCam"][frame_index]], dtype=np.float32),
                            "dot_num": dot_num,
                            "orientation": orientation,
                            "device_name": device_name,
                            "device_type": device_type,
                            "task": task,
                            "phase": phase,
                            "att_valid": att_valid,
                        }
                    )
                    if self.max_samples is not None and len(self.lines) >= self.max_samples:
                        return
        finally:
            if tar_obj is not None:
                tar_obj.close()

    def _build_apple_index(self, data_path):
        for subject, subject_path, info, device_name, device_type in self._subject_iter(data_path):
            required = ["appleFace.json", "appleLeftEye.json", "appleRightEye.json", "dotInfo.json"]
            if not all(os.path.exists(os.path.join(subject_path, name)) for name in required):
                continue
            face_json = read_json(os.path.join(subject_path, "appleFace.json"))
            left_json = read_json(os.path.join(subject_path, "appleLeftEye.json"))
            right_json = read_json(os.path.join(subject_path, "appleRightEye.json"))
            grid_path = os.path.join(subject_path, "faceGrid.json")
            grid_json = read_json(grid_path) if os.path.exists(grid_path) else None
            labels = read_json(os.path.join(subject_path, "dotInfo.json"))
            screen_path = os.path.join(subject_path, "screen.json")
            screen = read_json(screen_path) if os.path.exists(screen_path) else {}
            self.labels[subject] = labels
            self.screens[subject] = screen
            self.device_names[subject] = device_name
            self.device_types[subject] = device_type
            orientations = screen.get("Orientation", [])
            dot_nums = labels.get("DotNum", [])
            tasks = labels.get("Task", [])
            phases = labels.get("Phase", [])
            att_valids = labels.get("AttValid", [])
            for idx, _ in enumerate(face_json.get("X", [])):
                if not int(face_json["IsValid"][idx]) or not int(left_json["IsValid"][idx]) or not int(right_json["IsValid"][idx]):
                    continue
                if grid_json is not None and not int(grid_json["IsValid"][idx]):
                    continue
                orientation = int(orientations[idx]) if idx < len(orientations) else -1
                if self.orientation_filter and orientation not in self.orientation_filter:
                    continue
                face_x = int(face_json["X"][idx])
                face_y = int(face_json["Y"][idx])
                face_w = int(face_json["W"][idx])
                face_h = int(face_json["H"][idx])
                left_x = face_x + int(left_json["X"][idx])
                left_y = face_y + int(left_json["Y"][idx])
                left_w = int(left_json["W"][idx])
                left_h = int(left_json["H"][idx])
                right_x = face_x + int(right_json["X"][idx])
                right_y = face_y + int(right_json["Y"][idx])
                right_w = int(right_json["W"][idx])
                right_h = int(right_json["H"][idx])
                if min(face_w, face_h, left_w, left_h, right_w, right_h) <= 0:
                    continue
                if min(face_x, face_y, left_x, left_y, right_x, right_y) < 0:
                    continue
                frame = f"{idx:05d}.jpg"
                frame_path = os.path.join(subject_path, "frames", frame)
                if not os.path.exists(frame_path):
                    continue
                rects = np.array(
                    [
                        face_w,
                        face_h,
                        face_x,
                        face_y,
                        left_w,
                        left_h,
                        left_x,
                        left_y,
                        right_w,
                        right_h,
                        right_x,
                        right_y,
                    ],
                    dtype=np.float32,
                )
                dot_num = int(dot_nums[idx]) if idx < len(dot_nums) else -1
                task = str(tasks[idx]) if idx < len(tasks) else ""
                phase = str(phases[idx]) if idx < len(phases) else ""
                att_valid = int(att_valids[idx]) if idx < len(att_valids) else -1
                self.lines.append(
                    {
                        "subject": subject,
                        "frame": frame,
                        "frame_index": idx,
                        "frame_path": frame_path,
                        "boxes": [
                            [face_y, face_y + face_h, face_x, face_x + face_w],
                            [left_y, left_y + left_h, left_x, left_x + left_w],
                            [right_y, right_y + right_h, right_x, right_x + right_w],
                        ],
                        "rects": rects,
                        "rects_normalized": False,
                        "label": np.array([labels["XCam"][idx], labels["YCam"][idx]], dtype=np.float32),
                        "dot_num": dot_num,
                        "orientation": orientation,
                        "device_name": device_name,
                        "device_type": device_type,
                        "task": task,
                        "phase": phase,
                        "att_valid": att_valid,
                    }
                )
                if self.max_samples is not None and len(self.lines) >= self.max_samples:
                    return

    def __len__(self):
        return len(self.lines)

    def __getitems__(self, indices):
        """Load a batch concurrently without duplicating the dataset index."""
        if self.loader_threads <= 1 or len(indices) <= 1:
            return [self[index] for index in indices]
        if self._loader_executor is None:
            self._loader_executor = ThreadPoolExecutor(
                max_workers=self.loader_threads,
                thread_name_prefix="mpcas_gaze-loader",
            )
        return list(self._loader_executor.map(self.__getitem__, indices))

    def __getitem__(self, idx):
        item = self.lines[idx]
        boxes = copy.deepcopy(item["boxes"])
        rects = np.asarray(item["rects"], dtype=np.float32).copy()
        face_img = None
        left_img = None
        right_img = None

        packed = self._load_cached_crops(item)
        if packed is not None:
            if self.crop_cache_mode == "lmdb_jpeg":
                face_cached, left_cached, right_cached = unpack_jpeg_cached_crops(packed)
            else:
                face_cached, left_cached, right_cached = unpack_cached_crops(packed)
            if self.crop_cache_decode_on_gpu:
                face_img = cached_crop_to_uint8(face_cached)
                left_img = cached_crop_to_uint8(left_cached)
                right_img = cached_crop_to_uint8(right_cached)
            else:
                brightness = self.brightness if self.augment else 0.0
                contrast = self.contrast if self.augment else 0.0
                face_img = cached_crop_to_float(face_cached, brightness=brightness, contrast=contrast)
                left_img = cached_crop_to_float(left_cached, brightness=brightness, contrast=contrast)
                right_img = cached_crop_to_float(right_cached, brightness=brightness, contrast=contrast)

        if face_img is None:
            image = self._load_frame_image(item)
            height, width = image.shape[:2]
            if not item["rects_normalized"]:
                rects = rects / np.array([width, height] * 6, dtype=np.float32)
                rects_normalized = True
            else:
                rects_normalized = True

            if self.augment:
                boxes, rects = jitter_boxes(boxes, rects, width, height, rects_normalized=rects_normalized)

            face_img = preprocess_crop(
                crop(image, boxes[0]),
                (224, 224),
                brightness=self.brightness if self.augment else 0.0,
                contrast=self.contrast if self.augment else 0.0,
            )
            left_img = preprocess_crop(
                crop(image, boxes[1]),
                (112, 112),
                brightness=self.brightness if self.augment else 0.0,
                contrast=self.contrast if self.augment else 0.0,
            )
            right_img = preprocess_crop(
                crop(image, boxes[2]),
                (112, 112),
                flip=True,
                brightness=self.brightness if self.augment else 0.0,
                contrast=self.contrast if self.augment else 0.0,
            )
        else:
            rects_normalized = True
        del rects_normalized

        aux = np.zeros(self.aux_dim, dtype=np.float32)
        if self.aux_dim > 0:
            values = self.aux_table.get((item["subject"], item["frame"]))
            if values is not None:
                aux[: min(len(values), self.aux_dim)] = values[: self.aux_dim]

        return {
            "faceImg": torch.from_numpy(face_img),
            "leftEyeImg": torch.from_numpy(left_img),
            "rightEyeImg": torch.from_numpy(right_img),
            "rects": torch.from_numpy(rects.astype(np.float32)).float(),
            "aux": torch.from_numpy(aux).float(),
            "label": torch.from_numpy(item["label"]).float(),
            "subject": item["subject"],
            "frame": item["frame"],
            "frameIndex": torch.tensor(item["frame_index"], dtype=torch.long),
            "dotNum": torch.tensor(item["dot_num"], dtype=torch.long),
            "orientation": torch.tensor(item["orientation"], dtype=torch.long),
            "deviceName": item["device_name"],
            "deviceType": item["device_type"],
            "task": item.get("task", ""),
            "phase": item.get("phase", ""),
            "attValid": torch.tensor(item.get("att_valid", -1), dtype=torch.long),
        }


class BlockShuffleSampler(Sampler):
    def __init__(self, data_source, block_size=8192, seed=0, shuffle_within_block=True):
        self.data_source = data_source
        self.block_size = max(int(block_size), 1)
        self.seed = int(seed)
        self.shuffle_within_block = bool(shuffle_within_block)
        self.epoch = 0

    def __iter__(self):
        length = len(self.data_source)
        blocks = [(start, min(start + self.block_size, length)) for start in range(0, length, self.block_size)]
        rng = random.Random(self.seed + self.epoch)
        rng.shuffle(blocks)
        self.epoch += 1

        for start, end in blocks:
            indices = list(range(start, end))
            if self.shuffle_within_block:
                rng.shuffle(indices)
            yield from indices

    def __len__(self):
        return len(self.data_source)


def make_dataloader(config, split, shuffle=False):
    data_config = config.get("data", {})
    split_config = data_config.get(split, {})
    common = {k: v for k, v in data_config.items() if k not in {"train", "val", "test"}}
    params = {**common, **split_config}
    dataset = CoGazeDataset(
        data_paths=params.get("path") or params.get("paths"),
        split=params.get("split", split),
        annotation_mode=params.get("annotation_mode", "auto"),
        label_path=params.get("label_path"),
        subject_list=params.get("subject_list"),
        device_filter=params.get("device_filter"),
        device_type_filter=params.get("device_type_filter"),
        orientation_filter=params.get("orientation_filter"),
        max_samples=params.get("max_samples"),
        iphone_only=bool(params.get("iphone_only", False)),
        aux_path=params.get("aux_path"),
        aux_dim=int(config.get("model", {}).get("aux_dim", 0)),
        augment=bool(params.get("augment", split == "train")),
        brightness=float(params.get("brightness", 0.0)),
        contrast=float(params.get("contrast", 0.0)),
        crop_cache_dir=params.get("crop_cache_dir"),
        crop_cache_required=bool(params.get("crop_cache_required", True)),
        crop_cache_decode_on_gpu=bool(params.get("crop_cache_decode_on_gpu", True)),
        frame_cache_dir=params.get("frame_cache_dir"),
        frame_cache_required=bool(params.get("frame_cache_required", True)),
        loader_threads=int(params.get("loader_threads", data_config.get("loader_threads", 1))),
    )
    cache_text = ""
    if dataset.crop_cache_dir:
        cache_text = f" crop_cache={dataset.crop_cache_dir}"
        if dataset.crop_cache_mode:
            cache_text += f"({dataset.crop_cache_mode})"
    if dataset.frame_cache_dir:
        cache_text += f" frame_cache={dataset.frame_cache_dir}"
    annotation_text = ",".join(
        f"{name}:{count}" for name, count in dataset.annotation_counts.items() if count
    ) or "unknown"
    device_type_text = ",".join(
        f"{name}:{count}" for name, count in sorted(dataset.device_type_counts.items())
    ) or "none"
    print(
        f"[Read Data]: split={split} samples={len(dataset)} annotations={annotation_text} "
        f"device_types={device_type_text} paths={params.get('path') or params.get('paths')}{cache_text}"
    )
    num_workers = int(params.get("num_workers", data_config.get("num_workers", 0)))
    block_shuffle_size = int(params.get("block_shuffle_size", 0) or 0) if shuffle else 0
    sampler = None
    if block_shuffle_size > 0:
        sampler = BlockShuffleSampler(
            dataset,
            block_size=block_shuffle_size,
            seed=int(config.get("seed", 0)),
            shuffle_within_block=bool(params.get("block_shuffle_within_block", True)),
        )
        shuffle = False

    loader_kwargs = {
        "dataset": dataset,
        "batch_size": int(params.get("batch_size", data_config.get("batch_size", 128))),
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if sampler is not None:
        loader_kwargs["sampler"] = sampler
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(params.get("persistent_workers", True))
        loader_kwargs["prefetch_factor"] = int(params.get("prefetch_factor", 2))
    shuffle_text = ""
    if sampler is not None:
        shuffle_text = (
            f" block_shuffle_size={block_shuffle_size} "
            f"within_block={sampler.shuffle_within_block}"
        )
    print(
        f"[DataLoader]: split={split} batch_size={loader_kwargs['batch_size']} "
        f"num_workers={num_workers} pin_memory={loader_kwargs['pin_memory']} "
        f"loader_threads={dataset.loader_threads} "
        f"decode_on_gpu={dataset.crop_cache_decode_on_gpu}{shuffle_text}"
    )
    return DataLoader(**loader_kwargs)
