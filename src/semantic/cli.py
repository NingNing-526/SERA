"""Offline generation CLI and orchestration. All flags/defaults remain unchanged."""

import argparse
import json
import os
import pickle
import re
import shutil
import torch
from transformers import (
    AutoModelForMultimodalLM,
    AutoProcessor,
    BitsAndBytesConfig,
)
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
from semantic.checkpoints import (
    read_json_checkpoint,
    atomic_write_json,
)
from semantic.inputs import (
    load_inputs,
    parse_sample_id,
    segment_sort_key,
    build_text_contexts,
)
from semantic.cleaning import (
    clean_model_response,
    validate_and_normalize_description,
    empty_observation_description,
)
from semantic.storage import (
    load_or_initialize_output,
    ensure_raw_storage,
    ensure_fallback_storage,
    ensure_output_ids,
    reset_all_descriptions,
    reset_selected_descriptions,
    sanitize_existing_descriptions,
    has_modality_descriptions,
    has_any_description,
    reset_invalid_vision_descriptions,
    reset_empty_fallback_descriptions,
)
from semantic.generation import (
    generate_single_pass_description,
    generate,
    extract_video_frames,
    generate_from_messages,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-pkl', required=True)
    parser.add_argument('--media-manifest', required=True)
    parser.add_argument('--model-path', default=DEFAULT_MODEL_PATH)
    parser.add_argument('--output', required=True)
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=None,
        help='Optional global override for all generation stages.',
    )
    parser.add_argument('--text-max-new-tokens', type=int, default=512)
    parser.add_argument('--audio-max-new-tokens', type=int, default=128)
    parser.add_argument('--vision-max-new-tokens', type=int, default=256)
    parser.add_argument(
        '--reference-sampling',
        action='store_true',
        help=(
            'Use the sampling temperatures from the three supplied scripts. '
            'The default is deterministic generation because 4-bit Gemma may '
            'otherwise append multilingual loops or reasoning text.'
        ),
    )
    parser.add_argument('--max-generation-attempts', type=int, default=2)
    parser.add_argument(
        '--num-video-frames',
        type=int,
        default=4,
        help='Uniformly sampled image frames; the supplied video script uses 4.',
    )
    parser.add_argument(
        '--modalities',
        nargs='+',
        choices=('text', 'audio', 'vision'),
        default=('text', 'audio', 'vision'),
        help='Generate only the selected modalities.',
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        choices=('train', 'valid', 'test'),
        default=('train', 'valid', 'test'),
        help='Dataset splits to generate; useful for a prompt-quality pilot.',
    )
    parser.add_argument(
        '--max-samples-per-split',
        type=int,
        default=0,
        help='Generate at most this many samples per selected split; 0 means all.',
    )
    parser.add_argument(
        '--overwrite-existing',
        action='store_true',
        help=(
            'Regenerate every existing description. Use this when changing '
            'prompt versions; otherwise completed entries are resumed/skipped.'
        ),
    )
    parser.add_argument(
        '--overwrite-modalities',
        nargs='+',
        choices=('text', 'audio', 'vision'),
        default=(),
        help=(
            'Clear selected modalities across all splits before regeneration, '
            'preserving every other modality. This prevents prompt versions '
            'from being mixed when running a limited pilot.'
        ),
    )
    parser.add_argument(
        '--text-context-window',
        type=int,
        choices=(0, 1),
        default=0,
        help=(
            'Compatibility option for earlier experiments. The TEXT-reference '
            'prompt analyzes the current segment only, so keep this at 0.'
        ),
    )
    parser.add_argument(
        '--reset-empty-fallbacks',
        action='store_true',
        help=(
            'Regenerate only entries previously replaced by the safe empty '
            'fallback, while preserving all valid descriptions.'
        ),
    )
    parser.add_argument(
        '--sanitize-only',
        action='store_true',
        help=(
            'Rebuild rule-cleaned descriptions from saved _raw decoder responses '
            'and exit without loading Gemma.'
        ),
    )
    parser.add_argument(
        '--load-in-4bit',
        action='store_true',
        help='Load model weights in 4-bit with bitsandbytes.',
    )
    parser.add_argument(
        '--max-gpu-memory',
        default=None,
        help='Per-visible-GPU model memory limit, for example 12GiB.',
    )
    parser.add_argument(
        '--max-cpu-memory',
        default='120GiB',
        help='CPU memory available for model offload.',
    )
    parser.add_argument(
        '--offload-folder',
        default=(
            'data/gemma_offload'
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_samples_per_split < 0:
        raise ValueError('--max-samples-per-split cannot be negative')
    if args.text_context_window != 0:
        raise ValueError(
            'The supplied TEXT text prompt is segment-only. Use '
            '--text-context-window 0 for this strict reference configuration.'
        )
    requested_modalities = tuple(dict.fromkeys(args.modalities))
    overwrite_modalities = tuple(dict.fromkeys(args.overwrite_modalities))
    if not set(overwrite_modalities).issubset(requested_modalities):
        raise ValueError('--overwrite-modalities must be included in --modalities')
    dataset, manifest = load_inputs(args)
    output = load_or_initialize_output(args.output, dataset)
    previous_prompt_version = output['_meta'].get('prompt_version')
    current_sampling_mode = (
        'reference_sampling' if args.reference_sampling else 'deterministic'
    )
    print(
        f'generation config: prompt_version={PROMPT_VERSION}, '
        f'sampling_mode={current_sampling_mode}, '
        f'video_frames={args.num_video_frames}'
    )
    previous_sampling_mode = output['_meta'].get('sampling_mode')
    incompatible_output = (
        has_any_description(output)
        and previous_prompt_version != PROMPT_VERSION
        and not args.overwrite_existing
        and not overwrite_modalities
    )
    if incompatible_output:
        version_label = previous_prompt_version or 'legacy/unknown'
        raise ValueError(
            f'Existing output uses prompt version {version_label!r}, while the '
            f'current version is {PROMPT_VERSION!r}. Use --overwrite-existing '
            'or, preferably, a new --output path so prompt experiments are not mixed.'
        )
    if (
        has_any_description(output)
        and previous_prompt_version == PROMPT_VERSION
        and previous_sampling_mode != current_sampling_mode
        and not args.overwrite_existing
    ):
        raise ValueError(
            f'Existing output uses sampling mode {previous_sampling_mode!r}, '
            f'but the current run uses {current_sampling_mode!r}. Use a new '
            '--output path or --overwrite-existing so generation policies are '
            'not mixed in one experiment.'
        )
    previous_versions = dict(
        output['_meta'].get('modality_prompt_versions', {})
    )
    # Backward compatibility for the earlier text-only version metadata.
    if 'text' not in previous_versions:
        legacy_text_version = output['_meta'].get('text_prompt_version')
        if legacy_text_version:
            previous_versions['text'] = legacy_text_version
    previous_context_window = output['_meta'].get('text_context_window')
    for modality in requested_modalities:
        configuration_changed = (
            previous_versions.get(modality)
            != MODALITY_PROMPT_VERSIONS[modality]
        )
        if modality == 'text':
            configuration_changed = (
                configuration_changed
                or previous_context_window != args.text_context_window
            )
        if (
            configuration_changed
            and has_modality_descriptions(output, modality)
            and not args.overwrite_existing
            and modality not in overwrite_modalities
        ):
            raise ValueError(
                f'Existing {modality} descriptions use a different single-pass '
                'prompt configuration. Add '
                f'--overwrite-modalities {modality} to regenerate that stream, '
                'or choose a new --output path.'
            )
    if args.overwrite_existing:
        reset_count = reset_all_descriptions(output)
        print(f'reset {reset_count} descriptions for prompt regeneration')
    elif overwrite_modalities:
        reset_count = reset_selected_descriptions(
            output,
            overwrite_modalities,
        )
        print(
            f'reset {reset_count} selected descriptions for prompt regeneration'
        )
    output['_meta'].update({
        'prompt_version': PROMPT_VERSION,
        'model_path': args.model_path,
        'generation_pipeline': 'single_pass_then_rule_cleaning',
        'sampling_mode': current_sampling_mode,
    })
    current_versions = dict(previous_versions)
    for modality in requested_modalities:
        current_versions[modality] = MODALITY_PROMPT_VERSIONS[modality]
    output['_meta']['modality_prompt_versions'] = current_versions
    if 'text' in requested_modalities:
        output['_meta']['text_context_window'] = args.text_context_window
    sanitized_count = sanitize_existing_descriptions(output)
    if sanitized_count:
        print(
            f'sanitized {sanitized_count} saved descriptions containing '
            'reasoning/channel artifacts'
        )
    reset_count = 0
    if 'vision' in requested_modalities:
        reset_count += reset_invalid_vision_descriptions(
            output, args.splits, args.max_samples_per_split
        )
    if args.reset_empty_fallbacks:
        empty_reset_count = reset_empty_fallback_descriptions(
            output,
            args.splits,
            requested_modalities,
            args.max_samples_per_split,
        )
        print(
            f'reset {empty_reset_count} safe empty descriptions for regeneration'
        )
        reset_count += empty_reset_count
    if (
        sanitized_count
        or reset_count
        or args.overwrite_existing
        or overwrite_modalities
    ):
        atomic_write_json(output, args.output)
    if reset_count:
        print(f'reset {reset_count} invalid/empty descriptions for regeneration')

    if args.sanitize_only:
        atomic_write_json(output, args.output)
        print(f'sanitized descriptions saved to {args.output}')
        return

    processor = AutoProcessor.from_pretrained(
        args.model_path, local_files_only=True
    )
    if (
        getattr(processor, 'tokenizer', None) is not None
        and processor.tokenizer.pad_token_id is None
    ):
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    if torch.cuda.is_available():
        # V100 (compute capability 7.x) has no native BF16 execution.
        major, _ = torch.cuda.get_device_capability(0)
        model_dtype = torch.bfloat16 if major >= 8 else torch.float16
    else:
        model_dtype = torch.float32

    load_kwargs = {
        'local_files_only': True,
        'device_map': 'auto',
        'dtype': model_dtype,
        'offload_folder': args.offload_folder,
        'offload_state_dict': True,
    }
    if args.max_gpu_memory:
        load_kwargs['max_memory'] = {
            **{
                index: args.max_gpu_memory
                for index in range(torch.cuda.device_count())
            },
            'cpu': args.max_cpu_memory,
        }
    if args.load_in_4bit:
        load_kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=model_dtype,
            bnb_4bit_quant_type='nf4',
        )

    os.makedirs(args.offload_folder, exist_ok=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_path, **load_kwargs
    ).eval()
    device_map = getattr(model, 'hf_device_map', None)
    if device_map is None:
        device_map = {'model': str(model.device)}
    print(f'model dtype: {model_dtype}, device map: {device_map}')

    text_contexts = build_text_contexts(dataset, args.text_context_window)
    for split in args.splits:
        raw_texts = dataset[split]['raw_text']
        limit = args.max_samples_per_split or len(raw_texts)
        for index, raw_text in enumerate(raw_texts[:limit]):
            context = text_contexts[split][index]
            raw_requests = {
                'text': (
                    RAW_PROMPTS['text'].format(
                        text=str(raw_text),
                        previous_text=context['previous_text'],
                        next_text=context['next_text'],
                    ),
                    None,
                ),
                'audio': (
                    RAW_PROMPTS['audio'], manifest[split]['audio'][index]
                ),
                'vision': (
                    RAW_PROMPTS['vision'], manifest[split]['vision'][index]
                ),
            }
            for modality in requested_modalities:
                if output[split][modality][index]:
                    continue
                raw_prompt, media_path = raw_requests[modality]
                raw_response = output['_raw'][split][modality][index]
                cleaned_description = None
                is_fallback = False
                if raw_response:
                    cleaned_description = validate_and_normalize_description(
                        raw_response, modality
                    )
                if cleaned_description is None:
                    raw_response, cleaned_description, is_fallback = (
                        generate_single_pass_description(
                            model,
                            processor,
                            modality,
                            raw_prompt,
                            media_path,
                            args,
                        )
                    )
                output['_raw'][split][modality][index] = raw_response
                output[split][modality][index] = cleaned_description
                output['_fallback'][split][modality][index] = is_fallback
                atomic_write_json(output, args.output)
                print(
                    f'generated raw response {split}[{index}].{modality}: '
                    f'{raw_response}'
                )
                print(
                    f'rule-cleaned {split}[{index}].{modality}: '
                    f'{cleaned_description}'
                )

    print(f'saved descriptions to {args.output}')
