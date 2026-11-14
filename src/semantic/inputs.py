"""Input files, exact sample-ID alignment and within-split text context."""

import os
import json
import pickle


def load_inputs(args):
    with open(args.input_pkl, 'rb') as file:
        dataset = pickle.load(file)
    with open(args.media_manifest, 'r', encoding='utf-8') as file:
        manifest = json.load(file)

    for split in ('train', 'valid', 'test'):
        size = len(dataset[split]['raw_text'])
        expected_ids = [str(item) for item in dataset[split]['id']]
        manifest_ids = manifest[split].get('ids')
        if manifest_ids is None:
            raise KeyError(
                f'{split}.ids is missing from the media manifest. Rebuild it '
                'with scripts/build_mosi_media_manifest.py so sample order can '
                'be verified.'
            )
        manifest_ids = [str(item) for item in manifest_ids]
        if len(manifest_ids) != size:
            raise ValueError(
                f'{split}.ids has {len(manifest_ids)} entries, expected {size}'
            )
        for index, (expected_id, manifest_id) in enumerate(
            zip(expected_ids, manifest_ids)
        ):
            if expected_id != manifest_id:
                raise ValueError(
                    f'Media manifest sample order mismatch at {split}[{index}]: '
                    f'pickle ID={expected_id!r}, manifest ID={manifest_id!r}. '
                    'Rebuild the manifest from the same pickle.'
                )
        for modality in ('audio', 'vision'):
            paths = manifest[split][modality]
            if len(paths) != size:
                raise ValueError(
                    f'{split}.{modality} has {len(paths)} paths, expected {size}'
                )
            missing = [path for path in paths if not os.path.isfile(path)]
            if missing:
                raise FileNotFoundError(
                    f'{split}.{modality} contains missing media, first: {missing[0]}'
                )
    return dataset, manifest


def parse_sample_id(sample_id):
    value = str(sample_id)
    for delimiter in ('$_$', '$*$'):
        if delimiter in value:
            video_id, segment_id = value.rsplit(delimiter, 1)
            return video_id, segment_id
    raise ValueError(f'Unsupported MOSI sample ID: {value!r}')


def segment_sort_key(segment_id):
    try:
        return 0, int(segment_id)
    except (TypeError, ValueError):
        return 1, str(segment_id)


def build_text_contexts(dataset, context_window):
    contexts = {}
    for split in ('train', 'valid', 'test'):
        size = len(dataset[split]['raw_text'])
        split_contexts = [
            {'previous_text': 'none', 'next_text': 'none'} for _ in range(size)
        ]
        if context_window == 0:
            contexts[split] = split_contexts
            continue
        grouped = {}
        for index, sample_id in enumerate(dataset[split]['id']):
            video_id, segment_id = parse_sample_id(sample_id)
            grouped.setdefault(video_id, []).append((segment_id, index))
        raw_texts = dataset[split]['raw_text']
        for samples in grouped.values():
            samples.sort(key=lambda item: segment_sort_key(item[0]))
            for position, (_, index) in enumerate(samples):
                if position > 0:
                    previous_index = samples[position - 1][1]
                    split_contexts[index]['previous_text'] = str(
                        raw_texts[previous_index]
                    )
                if position + 1 < len(samples):
                    next_index = samples[position + 1][1]
                    split_contexts[index]['next_text'] = str(
                        raw_texts[next_index]
                    )
        contexts[split] = split_contexts
    return contexts
