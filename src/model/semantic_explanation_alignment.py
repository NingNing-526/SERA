"""Cross-attention alignment between selected explanations and raw features."""

import torch
from torch import nn


class SemanticExplanationAlignment(nn.Module):
    """
    TEXT-style explanation alignment.

    Selected large-model semantic tokens are queries, while the corresponding
    original unimodal sequence supplies keys and values. The output keeps the
    explanation-token length so it can be appended to the original modality.
    """

    def __init__(
        self,
        hidden_dim,
        num_heads=8,
        dropout=0.1,
        gated=False,
        gate_init=0.1,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError('hidden_dim must be divisible by num_heads')
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.feature_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.gated = bool(gated)
        if self.gated:
            # Each modality owns an independent pair of scalar residual gates.
            # A small initial value keeps the module near the semantic-anchor
            # identity path while retaining a strong gradient for the gates.
            self.attention_scale = nn.Parameter(
                torch.tensor(float(gate_init))
            )
            self.ffn_scale = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(
        self,
        explanation,
        feature,
        explanation_valid=None,
        feature_padding_mask=None,
    ):
        if explanation.ndim != 3 or feature.ndim != 3:
            raise ValueError('explanation and feature must both be [B, L, D]')
        if explanation.size(0) != feature.size(0):
            raise ValueError('explanation and feature batch sizes must match')

        if explanation_valid is None:
            explanation_valid = torch.ones(
                explanation.shape[:2],
                dtype=torch.bool,
                device=explanation.device,
            )
        else:
            explanation_valid = explanation_valid.to(
                device=explanation.device, dtype=torch.bool
            )
        if explanation_valid.shape != explanation.shape[:2]:
            raise ValueError('explanation_valid shape does not match explanation')

        safe_feature_mask = None
        if feature_padding_mask is not None:
            safe_feature_mask = feature_padding_mask.to(
                device=feature.device, dtype=torch.bool
            ).clone()
            if safe_feature_mask.shape != feature.shape[:2]:
                raise ValueError(
                    'feature_padding_mask shape does not match feature sequence'
                )
            # MultiheadAttention produces NaNs when every key is masked. Keep a
            # zero-valued key available for such defensive edge cases.
            all_masked = safe_feature_mask.all(dim=1)
            if all_masked.any():
                safe_feature_mask[all_masked, 0] = False

        query = self.query_norm(explanation)
        key_value = self.feature_norm(feature)
        attended, _ = self.attention(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=safe_feature_mask,
            need_weights=False,
        )
        if self.gated:
            attention_scale = torch.tanh(self.attention_scale)
            aligned = self.attention_norm(
                explanation + attention_scale * self.dropout(attended)
            )
            ffn_scale = torch.tanh(self.ffn_scale)
            aligned = self.output_norm(
                aligned + ffn_scale * self.ffn(aligned)
            )
        else:
            # Preserve the original implementation exactly for legacy runs and
            # checkpoints when gated alignment is disabled.
            aligned = self.attention_norm(
                explanation + self.dropout(attended)
            )
            aligned = self.output_norm(aligned + self.ffn(aligned))
        return aligned.masked_fill(~explanation_valid.unsqueeze(-1), 0.0)
