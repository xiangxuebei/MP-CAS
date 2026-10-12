import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import yaml

try:
    from torch.cuda.amp import GradScaler, autocast
except ImportError:  # pragma: no cover
    GradScaler = None
    autocast = None

try:
    from .dataset import image_batch_to_float, make_dataloader
    from .model import build_model_from_config, strip_module_prefix
except ImportError:  # Allow direct script execution.
    from dataset import image_batch_to_float, make_dataloader
    from model import build_model_from_config, strip_module_prefix


def parse_args():
    parser = argparse.ArgumentParser(description="Train CoGazeNet.")
    parser.add_argument("--config", default="configs/mpcas_gaze_example.yaml")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--val-data-path", default=None)
    parser.add_argument("--val-split", default=None)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--init-checkpoint", default=None, help="Load model weights only; keep a fresh optimizer/scheduler.")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--device-type",
        default=None,
        choices=["phone", "tablet", "unknown"],
        help="Filter train/val/test samples by classified device type.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as infile:
        return yaml.safe_load(infile) or {}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def override_config(config, args):
    if args.data_path:
        config.setdefault("data", {}).setdefault("train", {})["path"] = args.data_path
        config.setdefault("data", {}).setdefault("test", {})["path"] = args.data_path
    if args.val_data_path:
        config.setdefault("data", {}).setdefault("val", {})["path"] = args.val_data_path
    if args.val_split:
        config.setdefault("data", {}).setdefault("val", {})["split"] = args.val_split
    if args.batch_size:
        config.setdefault("data", {})["batch_size"] = args.batch_size
    if args.device_type:
        data_config = config.setdefault("data", {})
        data_config["device_type_filter"] = args.device_type
        for split in ("train", "val", "test"):
            data_config.setdefault(split, {})["device_type_filter"] = args.device_type
    if args.max_samples:
        config.setdefault("data", {}).setdefault("train", {})["max_samples"] = args.max_samples
    if args.epochs:
        config.setdefault("train", {})["epochs"] = args.epochs
    if args.lr:
        config.setdefault("optimizer", {})["lr"] = args.lr
    if args.save_dir:
        config.setdefault("train", {})["save_dir"] = args.save_dir
    if args.log_interval is not None:
        config.setdefault("train", {})["log_interval"] = args.log_interval
    return config


def format_duration(seconds):
    seconds = max(float(seconds), 0.0)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{sec:02d}s"


def ensure_validation_config(config):
    data_config = config.setdefault("data", {})
    val_config = data_config.get("val")
    train_config = data_config.get("train") or {}
    test_config = data_config.get("test") or {}
    if not val_config or not val_config.get("path"):
        raise ValueError("Configure an independent data.val.path before training.")
    if str(val_config.get("split", "val")).lower() == "test":
        raise ValueError("data.val.split cannot be the test split.")
    if (val_config.get("path"), val_config.get("split", "val")) == (
        train_config.get("path"), train_config.get("split", "train")
    ):
        raise ValueError("Training and validation data must use separate splits.")
    if (val_config.get("path"), val_config.get("split", "val")) == (
        test_config.get("path"), test_config.get("split", "test")
    ):
        raise ValueError("Validation and test data must use separate splits.")
    return "val"


def describe_device(device, use_amp):
    if device.type == "cuda":
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(device_index)
        gpu_count = torch.cuda.device_count()
        print(f"[Device]: Using GPU training on cuda:{device_index} ({gpu_name}), gpu_count={gpu_count}, amp={use_amp}")
    else:
        print("[Device]: Using CPU training, amp=False")


def save_checkpoint(path, model, optimizer, scheduler, epoch, config, best_metric=None, scaler=None):
    state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(
        {
            "model_state": state,
            "optimizer_state": optimizer.state_dict() if optimizer else None,
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "scaler_state": scaler.state_dict() if scaler else None,
            "epoch": epoch,
            "best_metric": best_metric,
            "config": config,
        },
        path,
    )


def run_epoch(
    model,
    loader,
    device,
    loss_fn,
    optimizer=None,
    scaler=None,
    loss_scale=4.0,
    clip_grad=0.0,
    phase="train",
    epoch=1,
    epochs=1,
    log_interval=1,
    image_brightness=0.0,
    image_contrast=0.0,
):
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_error = 0.0
    total_count = 0
    start = time.time()
    step_time_sum = 0.0
    data_time_sum = 0.0
    num_batches = len(loader)
    loader_iter = iter(loader)

    for batch_idx in range(num_batches):
        step_start = time.time()
        data = next(loader_iter)
        data_loaded = time.time()
        face = image_batch_to_float(
            data["faceImg"],
            device,
            non_blocking=True,
            augment=train,
            brightness=image_brightness,
            contrast=image_contrast,
        )
        left = image_batch_to_float(
            data["leftEyeImg"],
            device,
            non_blocking=True,
            augment=train,
            brightness=image_brightness,
            contrast=image_contrast,
        )
        right = image_batch_to_float(
            data["rightEyeImg"],
            device,
            non_blocking=True,
            augment=train,
            brightness=image_brightness,
            contrast=image_contrast,
        )
        rects = data["rects"].to(device, non_blocking=True)
        aux = data["aux"].to(device, non_blocking=True)
        label = data["label"].to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            if scaler and autocast:
                with autocast():
                    pred = model(left, right, face, rects, aux=aux)
                    loss = loss_fn(pred, label) * loss_scale
            else:
                pred = model(left, right, face, rects, aux=aux)
                loss = loss_fn(pred, label) * loss_scale

            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler:
                    scaler.scale(loss).backward()
                    if clip_grad > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if clip_grad > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                    optimizer.step()

        with torch.no_grad():
            error = torch.linalg.norm(pred - label, dim=1).sum().item()
        batch_size = label.size(0)
        total_loss += loss.item() * batch_size
        total_error += error
        total_count += batch_size
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_end = time.time()
        step_time = step_end - step_start
        data_time = data_loaded - step_start
        step_time_sum += step_time
        data_time_sum += data_time

        should_log = log_interval > 0 and ((batch_idx + 1) % log_interval == 0 or batch_idx == 0 or batch_idx + 1 == num_batches)
        if should_log:
            elapsed = max(step_end - start, 1.0e-6)
            avg_step_time = step_time_sum / (batch_idx + 1)
            avg_data_time = data_time_sum / (batch_idx + 1)
            epoch_remaining_steps = num_batches - batch_idx - 1
            epoch_eta = epoch_remaining_steps * avg_step_time
            if train:
                total_remaining_steps = epoch_remaining_steps + (epochs - epoch) * num_batches
                total_eta = total_remaining_steps * avg_step_time
                eta_text = f"epoch_eta={format_duration(epoch_eta)} train_eta={format_duration(total_eta)}"
            else:
                eta_text = f"epoch_eta={format_duration(epoch_eta)}"
            print(
                f"[{phase}] epoch={epoch}/{epochs} step={batch_idx + 1}/{num_batches} "
                f"step_time={step_time:.3f}s avg_step={avg_step_time:.3f}s "
                f"data_time={data_time:.3f}s avg_data={avg_data_time:.3f}s "
                f"loss={total_loss / total_count:.5f} error_cm={total_error / total_count:.5f} "
                f"samples_per_s={total_count / elapsed:.1f} {eta_text}"
            )
    return {
        "loss": total_loss / total_count,
        "error": total_error / total_count,
        "avg_step_time": step_time_sum / max(num_batches, 1),
        "avg_data_time": data_time_sum / max(num_batches, 1),
    }


def main():
    args = parse_args()
    if args.resume and args.init_checkpoint:
        raise ValueError("Use either --resume for continuing a run or --init-checkpoint for fine-tuning weights, not both.")
    config = override_config(load_config(args.config), args)
    set_seed(int(config.get("seed", 42)))

    train_config = config.get("train", {})
    optimizer_config = config.get("optimizer", {})
    save_dir = train_config.get("save_dir", "outputs/mpcas_gaze/train")
    checkpoint_dir = os.path.join(save_dir, "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)
    validation_source = ensure_validation_config(config)
    with open(os.path.join(save_dir, "config.resolved.yaml"), "w", encoding="utf-8") as outfile:
        yaml.safe_dump(config, outfile, sort_keys=False)

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    train_loader = make_dataloader(config, "train", shuffle=True)
    val_loader = make_dataloader(config, "val", shuffle=False)
    data_config = config.get("data", {})
    train_data_config = {**{k: v for k, v in data_config.items() if k not in {"train", "val", "test"}}, **data_config.get("train", {})}
    train_augments_images = bool(train_data_config.get("augment", True))
    train_brightness = float(train_data_config.get("brightness", 0.0)) if train_augments_images else 0.0
    train_contrast = float(train_data_config.get("contrast", 0.0)) if train_augments_images else 0.0
    print(f"[Validation]: validating after every epoch with data.{validation_source}.")

    model = build_model_from_config(config)
    resume_payload = None
    start_epoch = 1
    if args.init_checkpoint:
        init_payload = torch.load(args.init_checkpoint, map_location="cpu")
        incompatible = model.load_state_dict(strip_module_prefix(init_payload), strict=False)
        missing = getattr(incompatible, "missing_keys", incompatible[0])
        unexpected = getattr(incompatible, "unexpected_keys", incompatible[1])
        print(
            f"Initialized model weights from: {args.init_checkpoint} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
    if args.resume:
        resume_payload = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(strip_module_prefix(resume_payload), strict=False)
        print(f"Loaded checkpoint: {args.resume}")
    if torch.cuda.device_count() > 1 and device.type == "cuda":
        model = nn.DataParallel(model)
    model.to(device)

    loss_name = train_config.get("loss", "smooth_l1").lower()
    if loss_name in {"smooth_l1", "huber"}:
        loss_fn = nn.SmoothL1Loss(beta=float(train_config.get("smooth_l1_beta", 1.0)))
    elif loss_name == "mse":
        loss_fn = nn.MSELoss()
    else:
        loss_fn = nn.L1Loss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_config.get("lr", 1.0e-3)),
        weight_decay=float(optimizer_config.get("weight_decay", 5.0e-4)),
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(step) for step in optimizer_config.get("milestones", [8])],
        gamma=float(optimizer_config.get("gamma", 0.1)),
    )
    use_amp = bool(train_config.get("amp", True)) and device.type == "cuda" and GradScaler is not None
    scaler = GradScaler() if use_amp else None
    describe_device(device, use_amp)

    best_metric = float("inf")
    if resume_payload is not None:
        if resume_payload.get("optimizer_state") is not None:
            optimizer.load_state_dict(resume_payload["optimizer_state"])
        if resume_payload.get("scheduler_state") is not None:
            scheduler.load_state_dict(resume_payload["scheduler_state"])
        if scaler is not None and resume_payload.get("scaler_state") is not None:
            scaler.load_state_dict(resume_payload["scaler_state"])
        best_metric = float(resume_payload.get("best_metric") or best_metric)
        start_epoch = int(resume_payload.get("epoch", 0)) + 1
        print(
            f"[Resume]: continuing from epoch {start_epoch}, "
            f"best_val={best_metric:.5f}, lr={scheduler.get_last_lr()[0]:.6g}"
        )
        resume_epoch = start_epoch - 1
        if resume_epoch > 0:
            resumed_checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{resume_epoch:03d}_mpcas_gaze.pt")
            if not os.path.exists(resumed_checkpoint_path):
                save_checkpoint(
                    resumed_checkpoint_path,
                    model,
                    optimizer,
                    scheduler,
                    resume_epoch,
                    config,
                    best_metric,
                    scaler=scaler,
                )
            best_checkpoint_path = os.path.join(checkpoint_dir, "best_mpcas_gaze.pt")
            if not os.path.exists(best_checkpoint_path):
                save_checkpoint(
                    best_checkpoint_path,
                    model,
                    optimizer,
                    scheduler,
                    resume_epoch,
                    config,
                    best_metric,
                    scaler=scaler,
                )

    log_path = os.path.join(save_dir, "train_log.jsonl")
    epochs = int(train_config.get("epochs", 12))
    log_interval = int(train_config.get("log_interval", 50))
    if start_epoch > epochs:
        print(f"[Resume]: checkpoint epoch is already >= requested epochs ({epochs}); nothing to train.")
        return

    for epoch in range(start_epoch, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            loss_fn,
            optimizer=optimizer,
            scaler=scaler,
            loss_scale=float(train_config.get("loss_scale", 4.0)),
            clip_grad=float(train_config.get("clip_grad", 0.0)),
            phase="train",
            epoch=epoch,
            epochs=epochs,
            log_interval=log_interval,
            image_brightness=train_brightness,
            image_contrast=train_contrast,
        )
        scheduler.step()

        print(f"[Validation]: epoch {epoch} started")
        with torch.no_grad():
            val_metrics = run_epoch(
                model,
                val_loader,
                device,
                loss_fn,
                optimizer=None,
                scaler=None,
                loss_scale=float(train_config.get("loss_scale", 4.0)),
                phase="val",
                epoch=epoch,
                epochs=epochs,
                log_interval=log_interval,
            )
        metric = val_metrics["error"]
        print(
            f"[Validation]: epoch={epoch} val_loss={val_metrics['loss']:.5f} "
            f"val_error_cm={val_metrics['error']:.5f}"
        )

        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics, "lr": scheduler.get_last_lr()[0]}
        with open(log_path, "a", encoding="utf-8") as outfile:
            outfile.write(json.dumps(record) + "\n")

        if metric < best_metric:
            best_metric = metric
            save_checkpoint(
                os.path.join(checkpoint_dir, "best_mpcas_gaze.pt"),
                model,
                optimizer,
                scheduler,
                epoch,
                config,
                best_metric,
                scaler=scaler,
            )
        checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch:03d}_mpcas_gaze.pt")
        save_checkpoint(checkpoint_path, model, optimizer, scheduler, epoch, config, best_metric, scaler=scaler)
        print(
            f"epoch={epoch} train_error={train_metrics['error']:.5f} "
            f"val_error={val_metrics['error']:.5f} best_val={best_metric:.5f}"
        )


if __name__ == "__main__":
    main()
