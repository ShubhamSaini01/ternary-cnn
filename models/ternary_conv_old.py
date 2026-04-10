"""
Ternary Conv2d layer using TWN (Ternary Weight Networks) approach.

Key ideas:
- Maintain full-precision "shadow" weights for gradient updates
- Quantize to {-1, 0, +1} during forward pass using threshold Δ
- Per-channel scaling factor α recovers magnitude information
- Straight-Through Estimator (STE) for backprop through quantization

Reference: "Ternary Weight Networks" (Li et al., 2016)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TernaryQuantize(torch.autograd.Function):
    """Quantize weights to {-α, 0, +α} with STE for backward pass."""

    @staticmethod
    def forward(ctx, weights):
        # Compute threshold: Δ = 0.7 * mean(|w|) per output channel
        # Shape: weights is (out_channels, in_channels, kH, kW)
        abs_weights = weights.abs()
        # Mean over all dims except out_channels (dim 0)
        delta = 0.7 * abs_weights.mean(dim=(1, 2, 3), keepdim=True)

        # Ternary assignment
        ternary = torch.zeros_like(weights)
        ternary[weights > delta] = 1.0
        ternary[weights < -delta] = -1.0

        # Compute per-channel scaling factor α
        # α = mean of |w| where the weight is non-zero in ternary
        # Compute per-channel scaling factor α (vectorized — no Python loop)
        mask = ternary != 0
        # Count non-zero weights per channel
        count = mask.sum(dim=(1, 2, 3), keepdim=True).clamp(min=1)
        # Sum of |w| where ternary is non-zero
        alpha = (abs_weights * mask.float()).sum(dim=(1, 2, 3), keepdim=True) / count

        # Save for backward (not that we need it — STE just passes gradients through)
        ctx.save_for_backward(weights, delta)

        return ternary, alpha

    @staticmethod
    def backward(ctx, grad_ternary, grad_alpha):
        # STE: pass gradients straight through as if quantization didn't happen
        # Optionally clip to [-1, 1] to prevent exploding gradients
        weights, delta = ctx.saved_tensors
        # Clip gradient for weights outside [-1, 1] (optional but helps stability)
        grad = grad_ternary.clone()
        grad[weights.abs() > 1.0] = 0
        return grad


class TernaryConv2d(nn.Module):
    """
    Drop-in replacement for nn.Conv2d with ternary weights.

    Maintains full-precision weights internally. During forward pass,
    weights are quantized to {-1, 0, +1} with a per-channel scale α.
    Effective weight = α * ternary_weight.

    The quantization is transparent to the optimizer — gradients flow
    through via STE as if weights were full-precision.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding

        # Full-precision shadow weights (what the optimizer sees)
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, *self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

        # Initialize with Kaiming normal (same as standard Conv2d)
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # Quantize weights to ternary + get scaling factor
        ternary_w, alpha = TernaryQuantize.apply(self.weight)

        # Effective weight = α * ternary (broadcasting handles this)
        scaled_w = alpha * ternary_w

        return F.conv2d(x, scaled_w, self.bias,
                        stride=self.stride, padding=self.padding)

    def extra_repr(self):
        return (f"in={self.in_channels}, out={self.out_channels}, "
                f"kernel={self.kernel_size}, stride={self.stride}, "
                f"padding={self.padding}, bias={self.bias is not None}")
