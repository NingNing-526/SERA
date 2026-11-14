"""Build an order-safe MOSI audio/video manifest from processed sample IDs."""

import argparse
import json
import os
import pickle


DEFAULT_INPUT = 'data/mosi/unaligned_50.pkl'
DEFAULT_ROOT = 'data/mosi/media'
DEFAULT_OUTPUT = 'data/mosi/mosi_media_manifest.json'
AUDIO_EXTENSIONS = {'.wav', '.flac', '.mp3', '.m4a', '.ogg'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-pkl', default=DEFAULT_INPUT)
    parser.add_argument(
        '--audio-root', default=os.path.join(DEFAULT_ROOT, 'raw_audio')
    )
    parser.add_argument(
        '--vision-root', default=os.path.join(DEFAULT_ROOT, 'raw_data')
    )
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    return parser.parse_args()


def parse_sample_id(sample_id):
    value = str(sample_id)
    for delimiter in ('$_$', '$*$'):
        if delimiter in value:
            video_id, segment_id = value.rsplit(delimiter, 1)
            return video_id, segment_id
    raise ValueError(f"Unsupported MOSI sample ID: {value!r}")


def find_segment_file(root, video_id, segment_id, extensions):
    sample_dir = os.path.join(root, video_id)
    if not os.path.isdir(sample_dir):
        return None, f'missing directory: {sample_dir}'

    candidates = []
    for name in os.listdir(sample_dir):
        path = os.path.join(sample_dir, name)
        stem, extension = os.path.splitext(name)
        if (
            os.path.isfile(path)
            and stem == segment_id
            and extension.lower() in extensions
        ):
            candidates.append(os.path.abspath(path))

    if len(candidates) == 1:
        return candidates[0], None
    if not candidates:
        return None, f'no segment {segment_id} under {sample_dir}'
    return None, f'ambiguous segment {segment_id}: {candidates}'


def main():
    args = parse_args()
    with open(args.input_pkl, 'rb') as file:
        dataset = pickle.load(file)

    manifest = {}
    errors = []
    for split in ('train', 'valid', 'test'):
        manifest[split] = {'ids': [], 'audio': [], 'vision': []}
        for index, sample_id in enumerate(dataset[split]['id']):
            video_id, segment_id = parse_sample_id(sample_id)
            audio_path, audio_error = find_segment_file(
                args.audio_root, video_id, segment_id, AUDIO_EXTENSIONS
            )
            video_path, video_error = find_segment_file(
                args.vision_root, video_id, segment_id, VIDEO_EXTENSIONS
            )
            if audio_error or video_error:
                errors.append({
                    'split': split,
                    'index': index,
                    'id': str(sample_id),
                    'audio_error': audio_error,
                    'vision_error': video_error,
                })
                continue
            manifest[split]['ids'].append(str(sample_id))
            manifest[split]['audio'].append(audio_path)
            manifest[split]['vision'].append(video_path)

    if errors:
        print(f'manifest was not written: {len(errors)} samples failed validation')
        for error in errors[:20]:
            print(json.dumps(error, ensure_ascii=False))
        raise SystemExit(1)

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    for split in ('train', 'valid', 'test'):
        print(
            f"{split}: ids={len(manifest[split]['ids'])}, "
            f"audio={len(manifest[split]['audio'])}, "
            f"vision={len(manifest[split]['vision'])}"
        )
    print(f'saved manifest to {output_path}')


if __name__ == '__main__':
    main()
