"""Score semantic candidates, gather hard Top-K and apply straight-through gates. Auxiliary pooling uses ALL valid candidates."""


import torch
from torch import nn


class MultimodalFastPathway(nn.Module):
    """
    FRSR Fast Pathway for semantic tokens.
    """
    def __init__(self, hidden_dim, top_k=4):
        super().__init__()
        if top_k < 1:
            raise ValueError('top_k must be at least 1')
        self.top_k=top_k
        self.net=nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim,1,1)
        )

    def forward(self, x, valid_mask=None):
        """
        Args:
            x: semantic token sequence, [B, L, D].
            valid_mask: True for real tokens and False for padding, [B, L].
        Returns:
            Top-k evidence, all token scores, a top-k valid mask, the
            saliency-weighted pooled feature used for auxiliary supervision,
            and the selected positions in the input token sequence.
        """
        if x.ndim != 3:
            raise ValueError(f"Expected [B, L, D], got {tuple(x.shape)}")
        if valid_mask is None:
            valid_mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        else:
            valid_mask = valid_mask.to(device=x.device, dtype=torch.bool)
        if valid_mask.shape != x.shape[:2]:
            raise ValueError(
                f"Mask shape {tuple(valid_mask.shape)} does not match {tuple(x.shape[:2])}"
            )

        # Do not let padding values influence neighboring valid tokens through
        # the kernel-size-3 context convolution.
        x = x.masked_fill(~valid_mask.unsqueeze(-1), 0.0)

        logits = self.net(x.transpose(1, 2)).squeeze(1)
        masked_logits = logits.masked_fill(~valid_mask, torch.finfo(logits.dtype).min)
        k = min(self.top_k, x.size(1))
        top_logits, indices = torch.topk(masked_logits, k, dim=1)
        top_valid = torch.gather(valid_mask, 1, indices)

        gather_indices = indices.unsqueeze(-1).expand(-1, -1, x.size(-1))
        hard_evidence = torch.gather(x, 1, gather_indices)
        hard_evidence = hard_evidence.masked_fill(
            ~top_valid.unsqueeze(-1), 0.0
        )

        # Hard Top-K is discrete, so its indices cannot receive gradients from
        # the final multimodal prediction. Use a straight-through saliency gate:
        # its forward value is exactly one (the selected token is unchanged),
        # while its backward derivative follows sigmoid(top_logits). Therefore
        # the main task directly trains the scores of currently selected tokens.
        selected_probabilities = torch.sigmoid(top_logits)
        straight_through_gate = (
            torch.ones_like(selected_probabilities)
            + selected_probabilities
            - selected_probabilities.detach()
        )
        straight_through_gate = (
            straight_through_gate * top_valid.to(x.dtype)
        )
        evidence = hard_evidence * straight_through_gate.unsqueeze(-1)

        # Give every valid candidate (including tokens currently outside Top-K)
        # a differentiable auxiliary training signal. This avoids training the
        # selector only through the already selected positions.
        soft_weights = torch.softmax(masked_logits, dim=1)
        soft_weights = soft_weights * valid_mask.to(x.dtype)
        soft_weights = soft_weights / soft_weights.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-6)
        pooled = torch.sum(x * soft_weights.unsqueeze(-1), dim=1)

        scores = torch.sigmoid(logits).masked_fill(~valid_mask, 0.0)
        return evidence, scores, top_valid, pooled, indices
