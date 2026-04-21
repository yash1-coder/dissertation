"""
Minimal Vision Mamba / Vim-Tiny implementation for EuroSAT classification.

This module keeps the original Linux/CUDA-backed Vision Mamba path while also
providing a stronger CPU-only branch that preserves the core Vim design using a
pure-PyTorch selective scan backend.
"""

from __future__ import annotations

import math
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

try:
    from mamba_ssm import Mamba as _Mamba
except Exception:
    try:
        from mamba_ssm.modules.mamba_simple import Mamba as _Mamba
    except Exception as exc:  # pragma: no cover - exercised only when deps are missing
        _Mamba = None
        _MAMBA_IMPORT_ERROR = exc
    else:
        _MAMBA_IMPORT_ERROR = None
else:
    _MAMBA_IMPORT_ERROR = None


class _MambaCPUFallback(nn.Module):
    """Lightweight interface-compatible fallback for local Vim smoke tests."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        layer_idx: int | None = None,
    ) -> None:
        super().__init__()
        d_inner = int(expand * d_model)
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, d_conv, padding=d_conv - 1, groups=d_inner, bias=True
        )
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, seq_len, _ = x.shape
        xz = self.in_proj(x)
        x_part, z = xz.chunk(2, dim=-1)
        x_part = self.conv1d(x_part.transpose(1, 2))[..., :seq_len].transpose(1, 2)
        return self.out_proj(F.silu(x_part) * F.silu(z))


_MAMBA_CPU_FALLBACK: bool = False
if _Mamba is None:
    _Mamba = _MambaCPUFallback
    _MAMBA_CPU_FALLBACK = True


def _to_2tuple(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, tuple):
        return value
    return (value, value)


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    return x * random_tensor.div(keep_prob)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """Minimal patch embedding adapted from the official Vim implementation."""

    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int] = 16,
        stride: int | tuple[int, int] | None = None,
        in_chans: int = 3,
        embed_dim: int = 192,
    ) -> None:
        super().__init__()
        img_size = _to_2tuple(img_size)
        patch_size = _to_2tuple(patch_size)
        stride = patch_size if stride is None else _to_2tuple(stride)
        self.img_size = img_size
        self.patch_size = patch_size
        self.stride = stride
        self.grid_size = (
            (img_size[0] - patch_size[0]) // stride[0] + 1,
            (img_size[1] - patch_size[1]) // stride[1] + 1,
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        if (height, width) != self.img_size:
            raise ValueError(
                f"Expected inputs of shape {self.img_size}, got {(height, width)}. "
                "Resize the images to the Vim input resolution before calling the model."
            )
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


@dataclass
class VimConfig:
    img_size: int = 224
    patch_size: int = 16
    stride: int = 16
    in_chans: int = 3
    num_classes: int = 1000
    embed_dim: int = 192
    depth: int = 24
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    use_cls_token: bool = True
    use_middle_cls_token: bool = True


@dataclass
class VimCPUFullConfig:
    img_size: int = 224
    patch_size: int = 16
    stride: int = 16
    in_chans: int = 3
    num_classes: int = 1000
    embed_dim: int = 128
    depth: int = 6
    d_state: int = 4
    d_conv: int = 4
    expand: int = 2
    mlp_ratio: float = 4.0
    drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    use_cls_token: bool = True
    use_middle_cls_token: bool = True


def vim_runtime_status(require_cuda: bool = True) -> tuple[bool, str]:
    """Return whether the local runtime is suitable for executing Vim."""

    if _MAMBA_CPU_FALLBACK:
        return (
            True,
            "Vim running with CPU fallback (mamba-ssm unavailable). "
            "Architecture is valid for local testing; full CUDA training requires "
            "`pip install --no-build-isolation -r requirements-vim.txt` on Linux.",
        )

    if platform.system() != "Linux":
        return (
            False,
            "The dissertation Vim pipeline is supported on Linux because upstream Mamba "
            "targets Linux-centric builds.",
        )

    if require_cuda and not torch.cuda.is_available():
        return (
            False,
            "The dissertation Vim pipeline targets NVIDIA CUDA because upstream Mamba "
            "is designed for GPU execution. Use a Linux/CUDA environment to train or evaluate Vim.",
        )

    return True, "Vim runtime looks available."


def _init_vim_style_weights(module: nn.Module, head: nn.Linear, pos_embed: nn.Parameter, cls_token: nn.Parameter | None) -> None:
    nn.init.trunc_normal_(pos_embed, std=0.02)
    if cls_token is not None:
        nn.init.trunc_normal_(cls_token, std=0.02)
    nn.init.trunc_normal_(head.weight, std=0.02)
    nn.init.zeros_(head.bias)

    for submodule in module.modules():
        if isinstance(submodule, nn.Linear) and submodule is not head:
            nn.init.trunc_normal_(submodule.weight, std=0.02)
            if submodule.bias is not None:
                nn.init.zeros_(submodule.bias)
        elif isinstance(submodule, (nn.Conv1d, nn.Conv2d)):
            nn.init.kaiming_normal_(submodule.weight, mode="fan_out")
            if submodule.bias is not None:
                nn.init.zeros_(submodule.bias)
        elif isinstance(submodule, nn.LayerNorm):
            nn.init.ones_(submodule.weight)
            nn.init.zeros_(submodule.bias)


class _CheckpointMixin:
    config: VimConfig | VimCPUFullConfig

    def checkpoint_payload(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = {
            "model_state_dict": self.state_dict(),
            "config": asdict(self.config),
        }
        if extra:
            payload.update(extra)
        return payload

    def save_checkpoint(self, path: str | Path, extra: dict[str, Any] | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_payload(extra=extra), path)
        return path


class _VisionMambaBase(_CheckpointMixin, nn.Module):
    def _setup_tokens(
        self,
        embed_dim: int,
        use_cls_token: bool,
        use_middle_cls_token: bool,
        num_patches: int,
        drop_rate: float,
    ) -> None:
        self.use_cls_token = use_cls_token
        self.use_middle_cls_token = use_middle_cls_token
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.num_tokens = 1
        else:
            self.cls_token = None
            self.num_tokens = 0

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

    def _insert_cls_token(self, x: torch.Tensor) -> tuple[torch.Tensor, int | None]:
        if not self.use_cls_token or self.cls_token is None:
            return x, None

        batch_size, seq_len, _ = x.shape
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        if self.use_middle_cls_token:
            token_position = seq_len // 2
            x = torch.cat((x[:, :token_position], cls_token, x[:, token_position:]), dim=1)
        else:
            token_position = 0
            x = torch.cat((cls_token, x), dim=1)
        return x, token_position


class BidirectionalMambaMixer(nn.Module):
    """Bidirectional sequence mixing built from two upstream Mamba blocks."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        layer_idx: int | None = None,
    ) -> None:
        super().__init__()
        self.forward_mamba = _Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            layer_idx=layer_idx,
        )
        self.backward_mamba = _Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            layer_idx=layer_idx,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_forward = self.forward_mamba(x)
        y_backward = torch.flip(self.backward_mamba(torch.flip(x, dims=[1])), dims=[1])
        return 0.5 * (y_forward + y_backward)


class VimBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        drop_path_prob: float = 0.0,
        layer_idx: int | None = None,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mixer = BidirectionalMambaMixer(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            layer_idx=layer_idx,
        )
        self.drop_path = DropPath(drop_path_prob) if drop_path_prob > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop_path(self.mixer(self.norm(x)))


class VisionMamba(_VisionMambaBase):
    """
    Minimal classification-oriented Vision Mamba implementation.

    The structure follows the official Vim recipe closely for EuroSAT use:
    patch embedding, absolute positional encoding, middle class token, stacked
    bidirectional Mamba blocks, final norm, and a linear classifier.
    """

    def __init__(self, config: VimConfig) -> None:
        super().__init__()
        self.config = config
        self.num_classes = config.num_classes
        self.embed_dim = config.embed_dim

        self.patch_embed = PatchEmbed(
            img_size=config.img_size,
            patch_size=config.patch_size,
            stride=config.stride,
            in_chans=config.in_chans,
            embed_dim=config.embed_dim,
        )
        self._setup_tokens(
            embed_dim=config.embed_dim,
            use_cls_token=config.use_cls_token,
            use_middle_cls_token=config.use_middle_cls_token,
            num_patches=self.patch_embed.num_patches,
            drop_rate=config.drop_rate,
        )

        dpr = torch.linspace(0, config.drop_path_rate, config.depth).tolist()
        self.layers = nn.ModuleList(
            [
                VimBlock(
                    dim=config.embed_dim,
                    d_state=config.d_state,
                    d_conv=config.d_conv,
                    expand=config.expand,
                    drop_path_prob=float(dpr[idx]),
                    layer_idx=idx,
                )
                for idx in range(config.depth)
            ]
        )
        self.norm = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.num_classes)

        _init_vim_style_weights(self, self.head, self.pos_embed, self.cls_token)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x, cls_position = self._insert_cls_token(x)
        x = self.pos_drop(x + self.pos_embed)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        if cls_position is None:
            return x.mean(dim=1)
        return x[:, cls_position, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))

    @classmethod
    def from_checkpoint(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "VisionMamba":
        payload = torch.load(path, map_location=map_location)
        config = VimConfig(**payload["config"])
        model = cls(config)
        state_dict = payload.get("model_state_dict", payload)
        model.load_state_dict(state_dict)
        return model


class CPUFullMLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        return self.drop2(x)


class CPUSelectiveScanMixer(nn.Module):
    """Pure-PyTorch selective scan mixer for CPU-friendly Vision Mamba blocks."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 8,
        d_conv: int = 4,
        expand: int = 2,
        layer_idx: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.dt_rank = max(8, math.ceil(d_model / 16))

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + (2 * d_state), bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        base = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(base).repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def _selective_scan(
        self,
        x_part: torch.Tensor,
        dt: torch.Tensor,
        b_term: torch.Tensor,
        c_term: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, inner_dim = x_part.shape
        state = x_part.new_zeros(batch_size, inner_dim, self.d_state)
        a = -torch.exp(self.A_log).unsqueeze(0)
        d_skip = self.D.view(1, inner_dim)
        outputs: list[torch.Tensor] = []

        for idx in range(seq_len):
            x_t = x_part[:, idx, :]
            dt_t = dt[:, idx, :].unsqueeze(-1)
            b_t = b_term[:, idx, :].unsqueeze(1)
            c_t = c_term[:, idx, :].unsqueeze(1)

            delta_a = torch.exp(dt_t * a)
            delta_b = (1.0 - delta_a) * b_t
            state = (delta_a * state) + (delta_b * x_t.unsqueeze(-1))
            y_t = (state * c_t).sum(dim=-1) + (d_skip * x_t)
            outputs.append(y_t.unsqueeze(1))

        return torch.cat(outputs, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, seq_len, _ = x.shape
        xz = self.in_proj(x)
        x_part, z = xz.chunk(2, dim=-1)
        x_part = self.conv1d(x_part.transpose(1, 2))[..., :seq_len].transpose(1, 2)
        x_part = F.silu(x_part)

        dt_bc = self.x_proj(x_part)
        dt_part, b_term, c_term = torch.split(dt_bc, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt_part)) + 1e-4
        b_term = torch.tanh(b_term)
        c_term = torch.tanh(c_term)

        y = self._selective_scan(x_part, dt, b_term, c_term)
        y = y * F.silu(z)
        return self.out_proj(y)


class BidirectionalCPUSelectiveMixer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 8,
        d_conv: int = 4,
        expand: int = 2,
        layer_idx: int | None = None,
    ) -> None:
        super().__init__()
        self.forward_mixer = CPUSelectiveScanMixer(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            layer_idx=layer_idx,
        )
        self.backward_mixer = CPUSelectiveScanMixer(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            layer_idx=layer_idx,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_forward = self.forward_mixer(x)
        y_backward = torch.flip(self.backward_mixer(torch.flip(x, dims=[1])), dims=[1])
        return 0.5 * (y_forward + y_backward)


class VimCPUFullBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        d_state: int = 8,
        d_conv: int = 4,
        expand: int = 2,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path_prob: float = 0.0,
        layer_idx: int | None = None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mixer = BidirectionalCPUSelectiveMixer(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            layer_idx=layer_idx,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = CPUFullMLP(dim=dim, mlp_ratio=mlp_ratio, drop=drop)
        self.drop_path = DropPath(drop_path_prob) if drop_path_prob > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionMambaCPUFull(_VisionMambaBase):
    """CPU-oriented Vision Mamba branch with pure-PyTorch selective scan blocks."""

    def __init__(self, config: VimCPUFullConfig) -> None:
        super().__init__()
        self.config = config
        self.num_classes = config.num_classes
        self.embed_dim = config.embed_dim

        self.patch_embed = PatchEmbed(
            img_size=config.img_size,
            patch_size=config.patch_size,
            stride=config.stride,
            in_chans=config.in_chans,
            embed_dim=config.embed_dim,
        )
        self._setup_tokens(
            embed_dim=config.embed_dim,
            use_cls_token=config.use_cls_token,
            use_middle_cls_token=config.use_middle_cls_token,
            num_patches=self.patch_embed.num_patches,
            drop_rate=config.drop_rate,
        )

        dpr = torch.linspace(0, config.drop_path_rate, config.depth).tolist()
        self.layers = nn.ModuleList(
            [
                VimCPUFullBlock(
                    dim=config.embed_dim,
                    d_state=config.d_state,
                    d_conv=config.d_conv,
                    expand=config.expand,
                    mlp_ratio=config.mlp_ratio,
                    drop=config.drop_rate,
                    drop_path_prob=float(dpr[idx]),
                    layer_idx=idx,
                )
                for idx in range(config.depth)
            ]
        )
        self.norm = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.num_classes)

        _init_vim_style_weights(self, self.head, self.pos_embed, self.cls_token)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x, cls_position = self._insert_cls_token(x)
        x = self.pos_drop(x + self.pos_embed)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        if cls_position is None:
            return x.mean(dim=1)
        return x[:, cls_position, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, map_location: str | torch.device = "cpu"
    ) -> "VisionMambaCPUFull":
        payload = torch.load(path, map_location=map_location)
        config = VimCPUFullConfig(**payload["config"])
        model = cls(config)
        state_dict = payload.get("model_state_dict", payload)
        model.load_state_dict(state_dict)
        return model


def vim_tiny_patch16_224(pretrained: bool = False, **kwargs: Any) -> VisionMamba:
    if pretrained:
        raise ValueError(
            "This dissertation implementation keeps Vim pretrained=False by default. "
            "Load a checkpoint explicitly if you want pretrained weights."
        )
    config = VimConfig(**kwargs)
    return VisionMamba(config)


def vim_tiny_cpu_full(pretrained: bool = False, **kwargs: Any) -> VisionMambaCPUFull:
    if pretrained:
        raise ValueError(
            "The CPU-full Vision Mamba branch does not provide pretrained weights. "
            "Load a checkpoint explicitly if you need saved parameters."
        )
    config = VimCPUFullConfig(**kwargs)
    return VisionMambaCPUFull(config)


__all__ = [
    "PatchEmbed",
    "VimConfig",
    "VimCPUFullConfig",
    "VisionMamba",
    "VisionMambaCPUFull",
    "vim_runtime_status",
    "vim_tiny_patch16_224",
    "vim_tiny_cpu_full",
]
