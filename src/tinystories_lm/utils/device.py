"""Single place where `cuda | mps | cpu` is resolved.

Nothing else in the repo is allowed to call `torch.cuda.is_available()`: all
device- and dtype-capability questions are answered here so that moving from a
laptop to an 8xH100 box is a config change, not a code change.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import ContextManager

import torch


@dataclass(frozen=True)
class DeviceInfo:
    """Everything the training loop needs to know about where it is running."""

    type: str  # "cuda" | "mps" | "cpu"
    device: torch.device
    supports_bf16: bool
    name: str

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.type} ({self.name}, bf16={'yes' if self.supports_bf16 else 'no'})"


def resolve_device(requested: str = "auto") -> DeviceInfo:
    """Map a config string to a concrete device.

    `auto` prefers cuda, then mps, then cpu. An explicit request for an
    unavailable device is an error rather than a silent downgrade -- silently
    training on CPU when you asked for CUDA is how a "why is this so slow" week
    starts.
    """
    requested = (requested or "auto").lower()
    have_cuda = torch.cuda.is_available()
    have_mps = torch.backends.mps.is_available()

    if requested == "auto":
        requested = "cuda" if have_cuda else ("mps" if have_mps else "cpu")

    if requested == "cuda":
        if not have_cuda:
            raise RuntimeError("device='cuda' requested but torch.cuda.is_available() is False")
        return DeviceInfo(
            type="cuda",
            device=torch.device("cuda"),
            supports_bf16=torch.cuda.is_bf16_supported(),
            name=torch.cuda.get_device_name(0),
        )
    if requested == "mps":
        if not have_mps:
            raise RuntimeError("device='mps' requested but torch.backends.mps.is_available() is False")
        # MPS accepts bf16 autocast on torch>=2.1, but kernel coverage is
        # patchier than CUDA's; `amp: auto` therefore leaves it off (see
        # `use_amp`) and you opt in with `amp: true`.
        return DeviceInfo(type="mps", device=torch.device("mps"), supports_bf16=True, name="Apple Silicon")
    if requested == "cpu":
        return DeviceInfo(type="cpu", device=torch.device("cpu"), supports_bf16=False, name="cpu")

    raise ValueError(f"unknown device {requested!r}; expected auto|cuda|mps|cpu")


def device_info_for(obj: torch.nn.Module | torch.Tensor | torch.device) -> DeviceInfo:
    """DeviceInfo for wherever a model/tensor already lives.

    Inference helpers use this instead of re-resolving `auto`: a model on CPU
    must not have its inputs placed on the GPU just because one is present.
    """
    if isinstance(obj, torch.nn.Module):
        device = next(obj.parameters()).device
    elif isinstance(obj, torch.Tensor):
        device = obj.device
    else:
        device = torch.device(obj)
    return resolve_device(device.type)


def use_amp(info: DeviceInfo, amp: str | bool = "auto") -> bool:
    """Decide whether to run under bf16 autocast.

    `auto` means "yes on CUDA if the card supports bf16, no otherwise". CPU bf16
    autocast is slower than fp32 for models this small, and MPS bf16 is opt-in
    (see above), so `auto` is conservative and the config can always force it.
    """
    if isinstance(amp, bool):
        if amp and not info.supports_bf16:
            raise RuntimeError(f"amp=true requested but {info.type} does not support bf16 here")
        return amp
    amp = str(amp).strip().lower()
    if amp == "auto":
        return info.type == "cuda" and info.supports_bf16
    if amp in ("true", "false"):
        return use_amp(info, amp == "true")
    raise ValueError(f"unknown amp setting {amp!r}; expected true|false|auto")


def autocast_ctx(info: DeviceInfo, enabled: bool) -> ContextManager:
    """bf16 autocast context for this device (or a no-op)."""
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=info.type, dtype=torch.bfloat16)


def synchronize(info: DeviceInfo) -> None:
    """Make async accelerator work observable -- needed for honest timings."""
    if info.type == "cuda":
        torch.cuda.synchronize()
    elif info.type == "mps":
        torch.mps.synchronize()
