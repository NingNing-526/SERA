"""Legacy contrastive helpers retained for existing imports; not a new SERA loss."""

import torch
from pytorch_metric_learning.losses.ntxent_loss import NTXentLoss
from pytorch_metric_learning.utils import common_functions as c_f


def get_all_pairs_indices(labels, ref_labels=None):
    """
    Given a tensor of labels, this will return 4 tensors.
    The first 2 tensors are the indices which form all positive pairs
    The second 2 tensors are the indices which form all negative pairs
    """
    if ref_labels is None:
        ref_labels = labels
    labels1 = labels.unsqueeze(1)
    labels2 = ref_labels.unsqueeze(0)
    matches = (labels1 == labels2).byte()
    diffs = matches ^ 1
    if ref_labels is labels:
        matches.fill_diagonal_(0)
    a1_idx, p_idx = torch.where(matches)
    a2_idx, n_idx = torch.where(diffs)
    return a1_idx, p_idx, a2_idx, n_idx


def convert_to_pairs(indices_tuple, labels):
    """
    This returns anchor-positive and anchor-negative indices,
    regardless of what the input indices_tuple is
    Args:
        indices_tuple: tuple of tensors. Each tensor is 1d and specifies indices
                        within a batch
        labels: a tensor which has the label for each element in a batch
    """
    if indices_tuple is None:
        return get_all_pairs_indices(labels)
    elif len(indices_tuple) == 4:
        return indices_tuple
    else:
        a, p, n = indices_tuple
        return a, p, a, n


class cont_NTXentLoss(NTXentLoss):
    def __init__(self, temperature=0.1, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature
        self.add_to_recordable_attributes(list_of_names=["temperature"], is_stat=False)
        self.label = None

    def update_label(self, label):
        self.label = label

    def compute_loss(self, embeddings, labels, indices_tuple):
        indices_tuple = convert_to_pairs(indices_tuple, labels)
        if all(len(x) <= 1 for x in indices_tuple):
            return self.zero_losses()
        mat = self.distance(embeddings)
        return self.loss_method(mat, labels, indices_tuple)

    def _compute_loss(self, pos_pairs, neg_pairs, indices_tuple):
        a1, p, a2, n = indices_tuple
        weight = 1
        if self.label is not None:
            weight = abs(self.label[a2] - self.label[n])

        if len(a1) > 0 and len(a2) > 0:
            dtype = neg_pairs.dtype
            # if dealing with actual distances, use negative distances
            if not self.distance.is_inverted:
                pos_pairs = -pos_pairs
                neg_pairs = -neg_pairs

            # print(torch.sum(neg_pairs))
            if self.label is not None:
                neg_pairs = neg_pairs * weight / 2
            pos_pairs = pos_pairs.unsqueeze(1) / self.temperature
            neg_pairs = neg_pairs / self.temperature
            n_per_p = c_f.to_dtype(a2.unsqueeze(0) == a1.unsqueeze(1), dtype=dtype)
            neg_pairs = neg_pairs * n_per_p
            neg_pairs[n_per_p == 0] = c_f.neg_inf(dtype)

            max_val = torch.max(
                pos_pairs, torch.max(neg_pairs, dim=1, keepdim=True)[0]
            ).detach()
            numerator = torch.exp(pos_pairs - max_val).squeeze(1)

            denominator = torch.sum(torch.exp(neg_pairs - max_val), dim=1) + numerator
            log_exp = torch.log((numerator / denominator) + c_f.small_val(dtype))
            return {
                "loss": {
                    "losses": -log_exp,
                    "indices": (a1, p),
                    "reduction_type": "pos_pair",
                }
            }
        return self.zero_losses()
