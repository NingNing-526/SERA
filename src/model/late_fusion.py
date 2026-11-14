"""Capacity-controlled late fusion used by the -TGF ablation."""

import torch
from torch import nn


class LateFusionAdapter(nn.Module):
    """Project pooled text/audio/vision from ``3D`` back to ``D``.

    The projection starts as an exact equal-weight average.  It is trainable,
    so the no-text-guidance baseline can learn a global late-fusion mapping
    without receiving the full model's per-token text queries.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        if hidden_dim < 1:
            raise ValueError('hidden_dim must be positive')
        self.hidden_dim = hidden_dim
        self.projection = nn.Linear(hidden_dim * 3, hidden_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.projection.weight.zero_()
            identity = torch.eye(
                self.hidden_dim,
                dtype=self.projection.weight.dtype,
                device=self.projection.weight.device,
            ) / 3.0
            for index in range(3):
                start = index * self.hidden_dim
                self.projection.weight[:, start:start + self.hidden_dim].copy_(
                    identity
                )

    def forward(self, text, audio, vision):
        expected_shape = text.shape
        if text.ndim != 2:
            raise ValueError('pooled modality features must be [B, D]')
        if audio.shape != expected_shape or vision.shape != expected_shape:
            raise ValueError('text/audio/vision pooled shapes must match')
        if text.size(-1) != self.hidden_dim:
            raise ValueError(
                f'expected hidden dimension {self.hidden_dim}, '
                f'got {text.size(-1)}'
            )
        return self.projection(torch.cat((text, audio, vision), dim=-1))
