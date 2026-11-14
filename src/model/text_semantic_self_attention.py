"""Joint contextualization for concatenated text and semantic-anchor tokens."""

import torch
from torch import nn


class TextSemanticSelfAttention(nn.Module):
    """A configurable residual self-attention stack with mask support.

    The first layer keeps the original parameter names so that checkpoints
    from the one-layer TSA experiment remain loadable when ``num_layers=1``.
    Extra layers are stored separately and are only created when requested.
    """

    def __init__(self, hidden_dim, num_heads=8, num_layers=1, dropout=0.1):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError('hidden_dim must be divisible by num_heads')
        if num_layers < 1:
            raise ValueError('num_layers must be at least 1')

        self.input_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.additional_layers = nn.ModuleList(
            _TextSemanticSelfAttentionLayer(
                hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers - 1)
        )

    def forward(self, sequence, padding_mask=None):
        if sequence.ndim != 3:
            raise ValueError('sequence must have shape [B, L, D]')

        safe_padding_mask = None
        if padding_mask is not None:
            safe_padding_mask = padding_mask.to(
                device=sequence.device,
                dtype=torch.bool,
            ).clone()
            if safe_padding_mask.shape != sequence.shape[:2]:
                raise ValueError(
                    'padding_mask shape does not match the input sequence'
                )

            # MultiheadAttention returns NaNs if a sample masks every key.
            all_masked = safe_padding_mask.all(dim=1)
            if all_masked.any():
                safe_padding_mask[all_masked, 0] = False

        refined = self._forward_first_layer(sequence, safe_padding_mask)
        for layer in self.additional_layers:
            refined = layer(refined, safe_padding_mask)

        if padding_mask is not None:
            refined = refined.masked_fill(
                padding_mask.to(
                    device=sequence.device,
                    dtype=torch.bool,
                ).unsqueeze(-1),
                0.0,
            )
        return refined

    def _forward_first_layer(self, sequence, safe_padding_mask):
        normalized = self.input_norm(sequence)
        attended, _ = self.attention(
            query=normalized,
            key=normalized,
            value=normalized,
            key_padding_mask=safe_padding_mask,
            need_weights=False,
        )
        refined = self.attention_norm(
            sequence + self.attention_dropout(attended)
        )
        return self.output_norm(refined + self.ffn(refined))


class _TextSemanticSelfAttentionLayer(nn.Module):
    """One additional pre-normalized attention/FFN residual layer."""

    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, sequence, safe_padding_mask=None):
        normalized = self.input_norm(sequence)
        attended, _ = self.attention(
            query=normalized,
            key=normalized,
            value=normalized,
            key_padding_mask=safe_padding_mask,
            need_weights=False,
        )
        refined = self.attention_norm(
            sequence + self.attention_dropout(attended)
        )
        return self.output_norm(refined + self.ffn(refined))
