"""Merge independently generated modality shards into one aligned JSON file."""

import argparse
import copy
import json
import os
import tempfile


SPLITS = ('train', 'valid', 'test')
MODALITIES = ('text', 'audio', 'vision')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text-audio', required=True)
    parser.add_argument('--vision', required=True)
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def load_json(path):
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)


def require_same_ids(left, right):
    for split in SPLITS:
        left_ids = [str(item) for item in left['_ids'][split]]
        right_ids = [str(item) for item in right['_ids'][split]]
        if left_ids != right_ids:
            raise ValueError(
                f'Shard sample IDs differ for {split}; refusing to merge.'
            )


def validate_complete(output):
    for split in SPLITS:
        expected = len(output['_ids'][split])
        for modality in MODALITIES:
            values = output[split][modality]
            raw_values = output['_raw'][split][modality]
            fallback = output['_fallback'][split][modality]
            if not (
                len(values) == len(raw_values) == len(fallback) == expected
            ):
                raise ValueError(
                    f'Length mismatch for {split}.{modality}: expected '
                    f'{expected}, got values={len(values)}, raw={len(raw_values)}, '
                    f'fallback={len(fallback)}'
                )
            missing = [
                index for index, value in enumerate(values)
                if value is None or not str(value).strip()
            ]
            if missing:
                raise ValueError(
                    f'{split}.{modality} still has {len(missing)} missing '
                    f'descriptions; first index={missing[0]}'
                )


def atomic_dump(data, path):
    output_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=os.path.basename(output_path) + '.',
        suffix='.tmp',
        dir=os.path.dirname(output_path),
    )
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def main():
    args = parse_args()
    text_audio = load_json(args.text_audio)
    vision = load_json(args.vision)
    require_same_ids(text_audio, vision)

    output = copy.deepcopy(text_audio)
    for split in SPLITS:
        output[split]['vision'] = vision[split]['vision']
        output['_raw'][split]['vision'] = vision['_raw'][split]['vision']
        output['_fallback'][split]['vision'] = vision['_fallback'][split]['vision']

    text_audio_meta = text_audio.get('_meta', {})
    vision_meta = vision.get('_meta', {})
    if (
        text_audio_meta.get('prompt_version')
        != vision_meta.get('prompt_version')
    ):
        raise ValueError('Shard prompt versions differ; refusing to merge.')
    modality_versions = dict(
        text_audio_meta.get('modality_prompt_versions', {})
    )
    modality_versions.update(
        vision_meta.get('modality_prompt_versions', {})
    )
    output.setdefault('_meta', {})[
        'modality_prompt_versions'
    ] = modality_versions

    validate_complete(output)
    atomic_dump(output, args.output)

    print(f'saved merged descriptions to {os.path.abspath(args.output)}')
    for split in SPLITS:
        counts = {
            modality: sum(output['_fallback'][split][modality])
            for modality in MODALITIES
        }
        print(f'{split}: samples={len(output["_ids"][split])}, fallbacks={counts}')


if __name__ == '__main__':
    main()
