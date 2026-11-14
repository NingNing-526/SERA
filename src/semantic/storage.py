"""Description storage schema, resume behavior, fallback flags and resets."""

import os
import shutil
from semantic.prompts import (
    PROMPT_VERSION,
    MODALITY_PROMPT_VERSIONS,
    TEXT_RAW_SYSTEM,
    TONE_LABELS,
    TONE_ALIASES,
    RAW_PROMPTS,
    INVALID_VISION_RESPONSES,
    INVALID_META_RESPONSES,
    DEFAULT_MODEL_PATH,
)
from semantic.checkpoints import read_json_checkpoint, atomic_write_json
from semantic.cleaning import validate_and_normalize_description, empty_observation_description


def load_or_initialize_output(path, dataset):
    if os.path.isfile(path):
        output = read_json_checkpoint(path)
        recovered_from = None
        if output is None:
            for recovery_path in (path, path + '.tmp', path + '.bak'):
                output = read_json_checkpoint(
                    recovery_path, repair_wrapping=True
                )
                if output is not None:
                    recovered_from = recovery_path
                    break
        if output is None:
            size = os.path.getsize(path)
            raise ValueError(
                f'Existing output JSON is empty or corrupted: {path} '
                f'({size} bytes). Automatic content repair and the .tmp/.bak '
                'checkpoints all failed. Move the broken file aside and '
                'regenerate it.'
            )
        if recovered_from is not None:
            corrupt_copy = path + '.corrupt'
            if recovered_from == path and not os.path.exists(corrupt_copy):
                shutil.copy2(path, corrupt_copy)
            print(
                f'warning: recovered invalid output {path} from '
                f'{recovered_from}'
            )
            atomic_write_json(output, path)
        output.setdefault('_meta', {})
        ensure_raw_storage(output, dataset)
        ensure_fallback_storage(output, dataset)
        ensure_output_ids(output, dataset)
        return output
    output = {
        split: {
            modality: [None] * len(dataset[split]['raw_text'])
            for modality in ('text', 'audio', 'vision')
        }
        for split in ('train', 'valid', 'test')
    }
    output['_meta'] = {}
    ensure_raw_storage(output, dataset)
    ensure_fallback_storage(output, dataset)
    ensure_output_ids(output, dataset)
    return output


def ensure_raw_storage(output, dataset):
    raw_output = output.setdefault('_raw', {})
    for split in ('train', 'valid', 'test'):
        split_output = raw_output.setdefault(split, {})
        expected = len(dataset[split]['raw_text'])
        for modality in ('text', 'audio', 'vision'):
            values = split_output.setdefault(modality, [None] * expected)
            if len(values) != expected:
                raise ValueError(
                    f'_raw.{split}.{modality} has {len(values)} entries; '
                    f'expected {expected}'
                )


def ensure_fallback_storage(output, dataset):
    """Store generation failure state separately from description content.

    A valid model prediction may be ``neutral``.  Therefore fallback state
    must never be inferred from the generated sentence itself.
    """
    had_fallback_storage = '_fallback' in output
    fallback_output = output.setdefault('_fallback', {})
    if not isinstance(fallback_output, dict):
        raise ValueError('_fallback must be a dictionary')

    for split in ('train', 'valid', 'test'):
        split_output = fallback_output.setdefault(split, {})
        if not isinstance(split_output, dict):
            raise ValueError(f'_fallback.{split} must be a dictionary')
        expected = len(dataset[split]['raw_text'])
        for modality in ('text', 'audio', 'vision'):
            flags = split_output.setdefault(modality, [False] * expected)
            if len(flags) != expected:
                raise ValueError(
                    f'_fallback.{split}.{modality} has {len(flags)} entries; '
                    f'expected {expected}'
                )
            if any(not isinstance(flag, bool) for flag in flags):
                raise ValueError(
                    f'_fallback.{split}.{modality} must contain booleans only'
                )

    if not had_fallback_storage and has_any_description(output):
        print(
            'warning: legacy description output has no fallback metadata. '
            'Existing rows are conservatively marked as non-fallback because '
            'their original generation status cannot be recovered from text. '
            'Use --overwrite-existing to regenerate them with tracked status.'
        )


def ensure_output_ids(output, dataset):
    """Bind every saved description row to its processed MOSI sample ID."""
    id_output = output.setdefault('_ids', {})
    for split in ('train', 'valid', 'test'):
        expected_ids = [str(item) for item in dataset[split]['id']]
        saved_ids = id_output.get(split)
        if saved_ids is None:
            id_output[split] = expected_ids
            if any(
                value
                for modality in ('text', 'audio', 'vision')
                for value in output[split][modality]
            ):
                print(
                    f'warning: legacy description output had no {split} IDs; '
                    'binding its existing positional order to the current pickle'
                )
            continue
        saved_ids = [str(item) for item in saved_ids]
        if len(saved_ids) != len(expected_ids):
            raise ValueError(
                f'Description ID count mismatch for {split}: '
                f'saved={len(saved_ids)}, current={len(expected_ids)}'
            )
        for index, (saved_id, expected_id) in enumerate(
            zip(saved_ids, expected_ids)
        ):
            if saved_id != expected_id:
                raise ValueError(
                    f'Description sample order mismatch at {split}[{index}]: '
                    f'saved ID={saved_id!r}, current ID={expected_id!r}. Use '
                    'the same processed pickle that generated this JSON.'
                )


def reset_all_descriptions(output):
    reset_count = 0
    for split in ('train', 'valid', 'test'):
        for modality in ('text', 'audio', 'vision'):
            for index, value in enumerate(output[split][modality]):
                if value:
                    output[split][modality][index] = None
                    reset_count += 1
                output['_raw'][split][modality][index] = None
                output['_fallback'][split][modality][index] = False
    return reset_count


def reset_selected_descriptions(output, modalities):
    reset_count = 0
    for split in ('train', 'valid', 'test'):
        for modality in modalities:
            values = output[split][modality]
            for index in range(len(values)):
                if values[index]:
                    values[index] = None
                    reset_count += 1
                output['_raw'][split][modality][index] = None
                output['_fallback'][split][modality][index] = False
    return reset_count


def sanitize_existing_descriptions(output):
    """Rebuild rule-cleaned descriptions without changing raw responses."""
    changed = 0
    for split in ('train', 'valid', 'test'):
        for modality in ('text', 'audio', 'vision'):
            cleaned_values = output[split][modality]
            raw_values = output['_raw'][split][modality]
            fallback_flags = output['_fallback'][split][modality]
            for index, raw_value in enumerate(raw_values):
                source = raw_value or cleaned_values[index]
                if not source:
                    continue
                repaired = validate_and_normalize_description(source, modality)
                # Never destroy a previously validated/safe value merely
                # because an older raw decoder response remains malformed.
                if repaired is None and cleaned_values[index]:
                    repaired = validate_and_normalize_description(
                        cleaned_values[index], modality
                    )
                if repaired is None:
                    continue
                if repaired != cleaned_values[index]:
                    cleaned_values[index] = repaired
                    changed += 1
                is_repaired_fallback = (
                    repaired == empty_observation_description(modality)
                )
                if fallback_flags[index] and not is_repaired_fallback:
                    fallback_flags[index] = False
                    changed += 1
    return changed


def has_modality_descriptions(output, modality):
    for split in ('train', 'valid', 'test'):
        if any(output[split][modality]):
            return True
    return False


def has_any_description(output):
    return any(
        value
        for split in ('train', 'valid', 'test')
        for modality in ('text', 'audio', 'vision')
        for value in output[split][modality]
    )


def reset_invalid_vision_descriptions(output, splits, max_samples):
    reset_count = 0
    for split in splits:
        values = output[split]['vision']
        limit = max_samples or len(values)
        for index, value in enumerate(values[:limit]):
            if not value:
                continue
            normalized = str(value).strip().lower()
            if any(pattern in normalized for pattern in INVALID_VISION_RESPONSES):
                output[split]['vision'][index] = None
                output['_raw'][split]['vision'][index] = None
                output['_fallback'][split]['vision'][index] = False
                reset_count += 1
    return reset_count


def reset_empty_fallback_descriptions(
    output, splits, modalities, max_samples
):
    reset_count = 0
    for split in splits:
        for modality in modalities:
            values = output[split][modality]
            fallback_flags = output['_fallback'][split][modality]
            limit = max_samples or len(values)
            for index, is_fallback in enumerate(fallback_flags[:limit]):
                if is_fallback:
                    output[split][modality][index] = None
                    output['_raw'][split][modality][index] = None
                    fallback_flags[index] = False
                    reset_count += 1
    return reset_count
