import argparse
from collections import OrderedDict

import torch
import torch.nn as nn


class SELayer(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        batch, channels, _, _ = x.shape
        scale = self.pool(x).view(batch, channels)
        scale = self.fc(scale).view(batch, channels, 1, 1)
        return x * scale


class AdaptiveGroupNorm(nn.Module):
    """AFF-Net style normalization guided by face/rect/aux features."""

    def __init__(self, factor_dim, channels, groups):
        super().__init__()
        self.norm = nn.GroupNorm(groups, channels, affine=False)
        self.fc = nn.Sequential(
            nn.Linear(factor_dim, channels * 2),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x, factor):
        style = self.fc(factor).view(-1, 2, x.size(1), 1, 1)
        x = self.norm(x)
        return x * (style[:, 0] + 1.0) + style[:, 1]


class LinearAdaptiveBatchNorm(nn.Module):
    """MGazeNet style LABN: BN followed by factor-conditioned shift/scale."""

    def __init__(self, factor_dim, channels):
        super().__init__()
        self.norm = nn.BatchNorm2d(channels, affine=False)
        self.fc = nn.Sequential(
            nn.Linear(factor_dim, channels * 2),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x, factor):
        style = self.fc(factor).view(-1, 2, x.size(1), 1, 1)
        x = self.norm(x)
        return x * (style[:, 0] + 1.0) + style[:, 1]


class PlainNorm(nn.Module):
    def __init__(self, channels, groups):
        super().__init__()
        self.norm = nn.GroupNorm(groups, channels)

    def forward(self, x, factor):
        del factor
        return self.norm(x)


def make_adaptive_norm(kind, factor_dim, channels, groups):
    kind = kind.lower()
    if kind == "adagn":
        return AdaptiveGroupNorm(factor_dim, channels, groups)
    if kind == "labn":
        return LinearAdaptiveBatchNorm(factor_dim, channels)
    if kind in {"gn", "plain"}:
        return PlainNorm(channels, groups)
    raise ValueError(f"Unsupported norm kind: {kind}")


class EyeBranch(nn.Module):
    def __init__(self, factor_dim=128, norm="labn", use_se=True):
        super().__init__()
        self.use_se = use_se
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm2d(24) if norm == "labn" else nn.GroupNorm(3, 24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 48, kernel_size=5, stride=1, padding=0),
        )
        self.norm48 = make_adaptive_norm(norm, factor_dim, 48, groups=6)
        self.block2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            SELayer(48, 16) if use_se else nn.Identity(),
            nn.Conv2d(48, 64, kernel_size=5, stride=1, padding=1),
        )
        self.norm64a = make_adaptive_norm(norm, factor_dim, 64, groups=8)
        self.down = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.norm128 = make_adaptive_norm(norm, factor_dim, 128, groups=16)
        self.block4 = nn.Sequential(
            nn.ReLU(inplace=True),
            SELayer(128, 16) if use_se else nn.Identity(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
        )
        self.norm64b = make_adaptive_norm(norm, factor_dim, 64, groups=8)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, factor):
        x = self.conv1(x)
        x = self.norm48(x, factor)
        x = self.block2(x)
        x = self.norm64a(x, factor)
        low = self.down(x)

        high = self.conv3(low)
        high = self.norm128(high, factor)
        high = self.block4(high)
        high = self.norm64b(high, factor)
        high = self.relu(high)
        return torch.cat((low, high), dim=1)


class FaceBranch(nn.Module):
    def __init__(self, norm="labn", use_se=True):
        super().__init__()
        norm48 = nn.BatchNorm2d(48) if norm == "labn" else nn.GroupNorm(6, 48)
        norm96 = nn.BatchNorm2d(96) if norm == "labn" else nn.GroupNorm(12, 96)
        norm128 = nn.BatchNorm2d(128) if norm == "labn" else nn.GroupNorm(16, 128)
        norm192 = nn.BatchNorm2d(192) if norm == "labn" else nn.GroupNorm(16, 192)
        norm128b = nn.BatchNorm2d(128) if norm == "labn" else nn.GroupNorm(16, 128)
        norm64 = nn.BatchNorm2d(64) if norm == "labn" else nn.GroupNorm(8, 64)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=5, stride=2, padding=0),
            norm48,
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 96, kernel_size=5, stride=1, padding=0),
            norm96,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 128, kernel_size=5, stride=1, padding=2),
            norm128,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1),
            norm192,
            nn.ReLU(inplace=True),
            SELayer(192, 16) if use_se else nn.Identity(),
            nn.Conv2d(192, 128, kernel_size=3, stride=2, padding=0),
            norm128b,
            nn.ReLU(inplace=True),
            SELayer(128, 16) if use_se else nn.Identity(),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=0),
            norm64,
            nn.ReLU(inplace=True),
            SELayer(64, 16) if use_se else nn.Identity(),
        )
        self.fc = nn.Sequential(
            nn.Linear(5 * 5 * 64, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x.flatten(1))


class CoGazeNet(nn.Module):
    """AFF/MGazeNet backbone with optional WebEyeTrack-style geometry features."""

    def __init__(
        self,
        norm="labn",
        use_se=True,
        aux_dim=0,
        aux_embed_dim=32,
        guidance_dim=128,
        dropout=0.0,
    ):
        super().__init__()
        self.aux_dim = int(aux_dim)
        self.aux_embed_dim = int(aux_embed_dim) if self.aux_dim > 0 else 0
        self.guidance_dim = int(guidance_dim)

        self.face_branch = FaceBranch(norm=norm, use_se=use_se)
        self.rect_fc = nn.Sequential(
            nn.Linear(12, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 96),
            nn.LeakyReLU(inplace=True),
            nn.Linear(96, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(inplace=True),
        )
        if self.aux_dim > 0:
            self.aux_fc = nn.Sequential(
                nn.Linear(self.aux_dim, self.aux_embed_dim),
                nn.LeakyReLU(inplace=True),
                nn.Linear(self.aux_embed_dim, self.aux_embed_dim),
                nn.LeakyReLU(inplace=True),
            )
        else:
            self.aux_fc = None

        raw_factor_dim = 64 + 64 + self.aux_embed_dim
        self.guidance_fc = nn.Sequential(
            nn.Linear(raw_factor_dim, self.guidance_dim),
            nn.LeakyReLU(inplace=True),
        )
        self.eye_branch = EyeBranch(
            factor_dim=self.guidance_dim,
            norm=norm,
            use_se=use_se,
        )
        self.eye_merge_conv = nn.Sequential(
            SELayer(256, 16) if use_se else nn.Identity(),
            nn.Conv2d(256, 64, kernel_size=3, stride=2, padding=1),
        )
        self.eye_merge_norm = make_adaptive_norm(norm, self.guidance_dim, 64, groups=8)
        self.eye_merge_tail = nn.Sequential(
            nn.ReLU(inplace=True),
            SELayer(64, 16) if use_se else nn.Identity(),
        )
        self.eye_fc = nn.Sequential(
            nn.Linear(5 * 5 * 64, 128),
            nn.LeakyReLU(inplace=True),
        )

        feature_dim = 128 + 64 + 64 + self.aux_embed_dim
        self.feature_dim = feature_dim
        self.gaze_head = nn.Sequential(
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(feature_dim, 128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(128, 2),
        )

    def forward(self, left_eye, right_eye, face, rects, aux=None, return_features=False):
        face_feature = self.face_branch(face)
        rect_feature = self.rect_fc(rects)
        factors = [face_feature, rect_feature]

        if self.aux_fc is not None:
            if aux is None:
                aux = torch.zeros(
                    left_eye.size(0),
                    self.aux_dim,
                    device=left_eye.device,
                    dtype=left_eye.dtype,
                )
            aux_feature = self.aux_fc(aux)
            factors.append(aux_feature)
        else:
            aux_feature = None

        raw_factor = torch.cat(factors, dim=1)
        guidance = self.guidance_fc(raw_factor)

        left_feature = self.eye_branch(left_eye, guidance)
        right_feature = self.eye_branch(right_eye, guidance)
        eye_feature = torch.cat((left_feature, right_feature), dim=1)
        eye_feature = self.eye_merge_conv(eye_feature)
        eye_feature = self.eye_merge_norm(eye_feature, guidance)
        eye_feature = self.eye_merge_tail(eye_feature)
        eye_feature = self.eye_fc(eye_feature.flatten(1))

        gaze_feature_parts = [eye_feature, face_feature, rect_feature]
        if aux_feature is not None:
            gaze_feature_parts.append(aux_feature)
        gaze_feature = torch.cat(gaze_feature_parts, dim=1)
        gaze = self.gaze_head(gaze_feature)

        if return_features:
            return {"gaze": gaze, "feature": gaze_feature, "guidance": guidance}
        return gaze


def strip_module_prefix(state_dict):
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if "model_state" in state_dict:
        state_dict = state_dict["model_state"]
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return OrderedDict(
        (key[7:] if key.startswith("module.") else key, value)
        for key, value in state_dict.items()
    )


def load_checkpoint(model, checkpoint_path, map_location="cpu", strict=True):
    payload = torch.load(checkpoint_path, map_location=map_location)
    state_dict = strip_module_prefix(payload)
    return model.load_state_dict(state_dict, strict=strict)


def build_model_from_config(config):
    model_config = config.get("model", {})
    return CoGazeNet(
        norm=model_config.get("norm", "labn"),
        use_se=bool(model_config.get("use_se", True)),
        aux_dim=int(model_config.get("aux_dim", 0)),
        aux_embed_dim=int(model_config.get("aux_embed_dim", 32)),
        guidance_dim=int(model_config.get("guidance_dim", 128)),
        dropout=float(model_config.get("dropout", 0.0)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--norm", default="labn", choices=["labn", "adagn", "gn"])
    parser.add_argument("--aux-dim", type=int, default=0)
    args = parser.parse_args()

    model = CoGazeNet(norm=args.norm, aux_dim=args.aux_dim)
    batch = 2
    aux = torch.zeros(batch, args.aux_dim) if args.aux_dim else None
    output = model(
        torch.zeros(batch, 3, 112, 112),
        torch.zeros(batch, 3, 112, 112),
        torch.zeros(batch, 3, 224, 224),
        torch.zeros(batch, 12),
        aux=aux,
        return_features=True,
    )
    params = sum(p.numel() for p in model.parameters())
    print({"gaze": tuple(output["gaze"].shape), "feature": tuple(output["feature"].shape), "params": params})


if __name__ == "__main__":
    main()

