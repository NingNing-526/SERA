"""Shared padding-mask construction for fixed-length multimodal features."""

import torch


def lengths_to_padding_mask(lengths, features, modality):
    """Return a CLS-aware padding mask using true sequence lengths.

    Args:
        lengths: one valid-frame count per sample.
        features: array/tensor shaped [N, L, D].
        modality: name used in validation errors.

    Returns:
        Boolean mask [N, L + 1], where True means padding. Column zero is
        reserved for the encoder's prepended CLS token and is always valid.
    """
    feature_shape = tuple(features.shape)
    if len(feature_shape) != 3:
        raise ValueError(
            f'{modality} features must be [N, L, D], got {feature_shape}'
        )
    num_samples, sequence_length, _ = feature_shape

    length_tensor = torch.as_tensor(lengths)
    if length_tensor.numel() != num_samples:
        raise ValueError(
            f'{modality}_lengths contains {length_tensor.numel()} values, '
            f'expected {num_samples}'
        )
    length_tensor = length_tensor.reshape(-1)
    if torch.is_floating_point(length_tensor):
        if not torch.isfinite(length_tensor).all():
            raise ValueError(f'{modality}_lengths contains NaN or infinity')
        if not torch.equal(length_tensor, length_tensor.round()):
            raise ValueError(f'{modality}_lengths must contain integer counts')
    length_tensor = length_tensor.to(dtype=torch.long)
    invalid = (length_tensor < 0) | (length_tensor > sequence_length)
    if invalid.any():
        index = int(invalid.nonzero(as_tuple=False)[0].item())
        raise ValueError(
            f'Invalid {modality}_lengths[{index}]={int(length_tensor[index])}; '
            f'expected a value in [0, {sequence_length}]'
        )

    positions = torch.arange(sequence_length).unsqueeze(0)
    feature_padding_mask = positions >= length_tensor.unsqueeze(1)
    cls_padding_mask = torch.zeros(num_samples, 1, dtype=torch.bool)
    return torch.cat((cls_padding_mask, feature_padding_mask), dim=1)
