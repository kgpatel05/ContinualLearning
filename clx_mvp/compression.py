# clx_mvp/compression.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from torch import Tensor


@dataclass
class LatentCompressionConfig:
    """
    Configuration for latent compression used by SIESTA and other latent replay flows.

    Attributes:
        method: compression type. Supported: "identity", "float16", "uniform".
        num_bits: bit width for uniform quantization (ignored for identity/float16).
        per_channel: if True, compute quantization scale per latent dimension.
        eps: numerical stability epsilon to avoid divide-by-zero.
        clamp: optional absolute value clamp before quantization.
    """
    method: str = "identity"
    num_bits: int = 8
    per_channel: bool = False
    eps: float = 1e-6
    clamp: Optional[float] = None


class LatentCompressor:
    """
    Base interface for latent compressors.
    """
    def compress(self, z: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Compress a batch of latents.
        Returns:
            encoded: Tensor (typically CPU) to be stored.
            aux: Optional Tensor with per-sample scales or metadata.
        """
        raise NotImplementedError

    def decompress(self, encoded: Tensor, aux: Optional[Tensor], device: Optional[torch.device] = None) -> Tensor:
        """
        Reconstruct a batch of latents given the encoded tensor and optional aux data.
        """
        raise NotImplementedError


class IdentityCompressor(LatentCompressor):
    """
    No-op compressor: stores full-precision latents on CPU.
    """
    def compress(self, z: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        return z.detach().cpu(), None

    def decompress(self, encoded: Tensor, aux: Optional[Tensor], device: Optional[torch.device] = None) -> Tensor:
        if device is None:
            return encoded
        return encoded.to(device)


class Float16Compressor(LatentCompressor):
    """
    Lightweight compression by casting latents to float16.
    """
    def compress(self, z: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        return z.detach().cpu().to(dtype=torch.float16), None

    def decompress(self, encoded: Tensor, aux: Optional[Tensor], device: Optional[torch.device] = None) -> Tensor:
        out = encoded.to(dtype=torch.float32)
        if device is not None:
            out = out.to(device)
        return out


class UniformQuantizationCompressor(LatentCompressor):
    """
    Symmetric uniform quantization with optional per-channel scaling.
    Stores int8/uint8-like tensors plus scale factors.
    """
    def __init__(self, num_bits: int = 8, per_channel: bool = False, eps: float = 1e-6, clamp: Optional[float] = None):
        self.num_bits = int(num_bits)
        self.per_channel = bool(per_channel)
        self.eps = float(eps)
        self.clamp = clamp
        self.qmax = 2 ** (self.num_bits - 1) - 1
        self.qmin = -self.qmax

    def compress(self, z: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        z_cpu = z.detach().cpu()
        if self.clamp is not None:
            z_cpu = torch.clamp(z_cpu, -self.clamp, self.clamp)

        if self.per_channel:
            # scale shape: [B, C] where C is flattened feature dim
            flat = z_cpu.view(z_cpu.size(0), -1)
            scale = flat.abs().amax(dim=1, keepdim=True) / float(self.qmax)
            scale = torch.clamp(scale, min=self.eps)
            q = torch.round(flat / scale).clamp(self.qmin, self.qmax).to(dtype=torch.int8)
            encoded = q.view_as(z_cpu)
        else:
            scale = z_cpu.abs().max() / float(self.qmax)
            scale = torch.clamp(scale, min=self.eps)
            encoded = torch.round(z_cpu / scale).clamp(self.qmin, self.qmax).to(dtype=torch.int8)
            # broadcast scale to per-batch for consistent shape
            scale = scale.expand(z_cpu.size(0))

        return encoded, scale

    def decompress(self, encoded: Tensor, aux: Optional[Tensor], device: Optional[torch.device] = None) -> Tensor:
        if aux is None:
            raise ValueError("Quantized latents require scale factors for decompression.")
        # aux may be scalar or per-sample vector; ensure broadcast
        if aux.dim() == 1:
            scale = aux.view(-1, *([1] * (encoded.dim() - 1)))
        else:
            scale = aux
        decoded = encoded.to(dtype=torch.float32) * scale.to(dtype=torch.float32)
        if device is not None:
            decoded = decoded.to(device)
        return decoded


def build_compressor(cfg: LatentCompressionConfig) -> LatentCompressor:
    """
    Factory for latent compressors.
    """
    method = cfg.method.lower()
    if method == "identity":
        return IdentityCompressor()
    if method in ("float16", "fp16"):
        return Float16Compressor()
    if method in ("uniform", "quant", "uniform-int8"):
        return UniformQuantizationCompressor(
            num_bits=cfg.num_bits,
            per_channel=cfg.per_channel,
            eps=cfg.eps,
            clamp=cfg.clamp,
        )
    raise ValueError(f"Unsupported compression method: {cfg.method}")
