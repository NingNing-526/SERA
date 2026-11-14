"""Precompute token-level semantic features before SERA training."""

import argparse
import json
import os
import pickle
import re
import sys

import torch


SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model.gemma_semantic_encoder import GemmaSemanticEncoder


MODALITIES = ('text', 'audio', 'vision')
SPLITS = ('train', 'valid', 'test')
DEFAULT_MODEL_PATH = os.environ.get(
    'GEMMA_MODEL_PATH',
    'models/gemma-4-E4B-it',
)

FIELD_LABEL_PATTERN = re.compile(
    r'\b(?:evidence|stance|affect|mechanism|target|cue|change|dynamics|'
    r'implied\s+affect|reliability|acoustic\s+evidence|inferred\s+tone|'
    r'confidence)\s*:\s*',
    flags=re.IGNORECASE,
)


def description_content(value, strip_field_labels, json_encoding):
    """Convert structured explanations to the requested encoder text."""
    text = str(value).strip()
    try:
        parsed = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        parsed = None
    if isinstance(parsed, dict):
        if strip_field_labels or json_encoding == 'values':
            return ' ; '.join(str(item).strip() for item in parsed.values())
        if json_encoding == 'naturalized':
            return ' ; '.join(
                f"{str(key).replace('_', ' ')}: {str(item).strip()}"
                for key, item in parsed.items()
            )
        return text
    if not strip_field_labels:
        return text
    return FIELD_LABEL_PATTERN.sub('', text).strip()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Encode offline LLM-enhanced text/audio/video descriptions as '
            'token-level features used by the FRSR Fast Pathway.'
        )
    )
    parser.add_argument('--input-pkl', required=True, help='MOSI processed pickle')
    parser.add_argument(
        '--descriptions-json',
        default=None,
        help=(
            'Optional JSON containing split -> modality -> list[str]. If omitted, '
            'the pickle must contain gemma_text_desc/gemma_audio_desc/'
            'gemma_vision_desc in every split.'
        ),
    )
    parser.add_argument(
        '--description-stage',
        choices=('raw', 'raw_response'),
        default='raw',
        help=(
            'Encode rule-cleaned single-pass raw descriptions (default), or the '
            'unmodified decoder responses stored under _raw for diagnostics.'
        ),
    )
    parser.add_argument(
        '--model-path',
        default=DEFAULT_MODEL_PATH,
        help=f'Local Gemma model path (default: {DEFAULT_MODEL_PATH})',
    )
    parser.add_argument('--output', required=True, help='Output .pt feature file')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-length', type=int, default=128)
    parser.add_argument(
        '--strip-field-labels',
        action='store_true',
        help=(
            'Deprecated compatibility mode: encode only JSON values or remove '
            'legacy field labels. Prefer --json-encoding naturalized.'
        ),
    )
    parser.add_argument(
        '--json-encoding',
        choices=('naturalized', 'values', 'full'),
        default='naturalized',
        help=(
            'How structured descriptions are presented to Gemma. '
            'naturalized keeps field-value meaning without JSON punctuation.'
        ),
    )
    parser.add_argument('--device', default=None, help='For example cuda:0 or cpu')
    parser.add_argument(
        '--load-in-4bit',
        action='store_true',
        help='Load Gemma weights in 4-bit with bitsandbytes.',
    )
    parser.add_argument(
        '--max-gpu-memory',
        default=None,
        help='Per-visible-GPU model memory limit, for example 16GiB.',
    )
    parser.add_argument(
        '--max-cpu-memory',
        default='120GiB',
        help='CPU memory available for model offload.',
    )
    parser.add_argument(
        '--offload-folder',
        default=(
            'data/gemma_token_offload'
        ),
    )
    return parser.parse_args()


def load_descriptions(args):
    with open(args.input_pkl, 'rb') as file:
        raw_data = pickle.load(file)

    if args.descriptions_json:
        with open(args.descriptions_json, 'r', encoding='utf-8') as file:
            loaded_descriptions = json.load(file)
        saved_ids = loaded_descriptions.get('_ids')
        if saved_ids is None:
            raise KeyError(
                'The description JSON has no _ids section. Run '
                'generate_semantic_descriptions.py --sanitize-only once with '
                'the same pickle and rebuilt media manifest to attach and '
                'validate sample IDs.'
            )
        for split in SPLITS:
            expected_ids = [str(item) for item in raw_data[split]['id']]
            split_ids = [str(item) for item in saved_ids.get(split, [])]
            if len(split_ids) != len(expected_ids):
                raise ValueError(
                    f'Description ID count mismatch for {split}: '
                    f'descriptions={len(split_ids)}, pickle={len(expected_ids)}'
                )
            for index, (description_id, expected_id) in enumerate(
                zip(split_ids, expected_ids)
            ):
                if description_id != expected_id:
                    raise ValueError(
                        f'Description sample order mismatch at {split}[{index}]: '
                        f'description ID={description_id!r}, '
                        f'pickle ID={expected_id!r}.'
                    )
        if args.description_stage == 'raw_response':
            if '_raw' not in loaded_descriptions:
                raise KeyError(
                    'The description file has no _raw decoder responses. '
                    'Regenerate it with the single-pass generation script.'
                )
            description_data = {
                split: loaded_descriptions['_raw'][split] for split in SPLITS
            }
            description_data['_meta'] = loaded_descriptions.get('_meta', {})
            description_data['_ids'] = saved_ids
        else:
            description_data = loaded_descriptions
    else:
        if args.description_stage == 'raw_response':
            raise ValueError(
                '--description-stage raw_response requires --descriptions-json '
                'from the single-pass generation script.'
            )
        description_data = {}
        for split in SPLITS:
            description_data[split] = {}
            for modality in MODALITIES:
                key = f'gemma_{modality}_desc'
                if key not in raw_data[split]:
                    raise KeyError(
                        f"Missing '{split}.{key}'. Generate the three modality "
                        "descriptions first or pass --descriptions-json."
                    )
                description_data[split][modality] = raw_data[split][key]

    for split in SPLITS:
        expected = len(raw_data[split]['raw_text'])
        for modality in MODALITIES:
            descriptions = description_data[split][modality]
            if len(descriptions) != expected:
                raise ValueError(
                    f"{split}.{modality} has {len(descriptions)} descriptions; "
                    f"the MOSI split has {expected} samples."
                )
            missing_indices = [
                index
                for index, item in enumerate(descriptions)
                if item is None or not str(item).strip()
            ]
            if missing_indices:
                raise ValueError(
                    f'{split}.{modality} contains {len(missing_indices)} '
                    'unfinished descriptions; first missing index: '
                    f'{missing_indices[0]}. Complete offline generation before '
                    'precomputing semantic tokens.'
                )
            normalized = [str(item) for item in descriptions]
            normalized = [
                description_content(
                    item, args.strip_field_labels, args.json_encoding
                )
                for item in normalized
            ]
            description_data[split][modality] = normalized
    return description_data


@torch.no_grad()
def encode_all(encoder, descriptions, batch_size, max_length):
    feature_batches = []
    mask_batches = []
    for start in range(0, len(descriptions), batch_size):
        batch = descriptions[start:start + batch_size]
        features, valid_mask = encoder(batch, max_length=max_length)
        feature_batches.append(features.detach().cpu().half())
        mask_batches.append(valid_mask.detach().cpu())
    features = torch.cat(feature_batches, dim=0)
    valid_mask = torch.cat(mask_batches, dim=0)
    # Gemma tokenizers may use left padding. Keep the complete column interval
    # containing real tokens instead of assuming they start at column zero.
    valid_columns = valid_mask.any(dim=0)
    if not valid_columns.any():
        raise RuntimeError('The encoded description set contains no valid tokens')
    valid_indices = valid_columns.nonzero(as_tuple=False).flatten()
    first_valid = int(valid_indices[0].item())
    last_valid = int(valid_indices[-1].item()) + 1
    return (
        features[:, first_valid:last_valid],
        valid_mask[:, first_valid:last_valid],
    )


def main():
    args = parse_args()
    descriptions = load_descriptions(args)
    with open(args.input_pkl, 'rb') as file:
        raw_data = pickle.load(file)
    encoder = GemmaSemanticEncoder(
        args.model_path,
        device=args.device,
        load_in_4bit=args.load_in_4bit,
        max_gpu_memory=args.max_gpu_memory,
        max_cpu_memory=args.max_cpu_memory,
        offload_folder=args.offload_folder,
    )

    output = {'metadata': {
        'input_pkl': os.path.abspath(args.input_pkl),
        'model_path': args.model_path,
        'max_length': args.max_length,
        'strip_field_labels': args.strip_field_labels,
        'json_encoding': (
            'values' if args.strip_field_labels else args.json_encoding
        ),
        'description_stage': args.description_stage,
        'description_prompt_version': descriptions.get('_meta', {}).get(
            'prompt_version'
        ),
        'text_prompt_version': descriptions.get('_meta', {}).get(
            'text_prompt_version'
        ),
        'text_context_window': descriptions.get('_meta', {}).get(
            'text_context_window'
        ),
        'generation_pipeline': descriptions.get('_meta', {}).get(
            'generation_pipeline'
        ),
        'modality_prompt_versions': descriptions.get('_meta', {}).get(
            'modality_prompt_versions', {}
        ),
    }}
    for split in SPLITS:
        output[split] = {
            # Preserve the exact positional identity used to generate every
            # feature row. The training loader rejects a different order.
            'ids': [str(item) for item in raw_data[split]['id']],
        }
        for modality in MODALITIES:
            features, mask = encode_all(
                encoder,
                descriptions[split][modality],
                args.batch_size,
                args.max_length,
            )
            output[split][modality] = features
            output[split][f'{modality}_mask'] = mask
            print(
                f'encoded {split}.{modality}: features={tuple(features.shape)}, '
                f'valid_tokens={int(mask.sum())}'
            )

    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    torch.save(output, args.output)
    print(f'saved offline semantic tokens to {args.output}')


if __name__ == '__main__':
    main()
