"""Media preparation, model calls and retry loop; generation policy unchanged."""

import torch
from semantic.prompts import (
    PROMPT_VERSION,
    MODALITY_PROMPT_VERSIONS,
    TEXT_RAW_SYSTEM,
    TEXT_RAW_EXAMPLE_USER,
    TEXT_RAW_EXAMPLE_ASSISTANT,
    TONE_LABELS,
    TONE_ALIASES,
    RAW_PROMPTS,
    INVALID_VISION_RESPONSES,
    INVALID_META_RESPONSES,
    DEFAULT_MODEL_PATH,
)
from semantic.cleaning import clean_model_response, validate_and_normalize_description, empty_observation_description


def generate_single_pass_description(
    model, processor, modality, prompt, media_path, args
):
    """Run Gemma once per attempt and rule-clean its decoded response."""
    last_value = None
    for attempt in range(max(args.max_generation_attempts, 1)):
        attempt_prompt = prompt
        if attempt:
            attempt_prompt += (
                ' Your previous response was empty, meta-level, or did not '
                'describe the supplied input. Return only the requested grounded '
                'natural-language explanation.'
            )
        last_value = generate(
            model,
            processor,
            modality,
            attempt_prompt,
            media_path,
            args,
            text_raw_few_shot=(modality == 'text'),
            generation_profile=f'raw_{modality}',
        )
        normalized = clean_model_response(last_value)
        lowered = normalized.lower()
        invalid_vision = (
            modality == 'vision'
            and any(item in lowered for item in INVALID_VISION_RESPONSES)
        )
        canonical = validate_and_normalize_description(normalized, modality)
        if (
            normalized
            and len(normalized.split()) <= 180
            and not invalid_vision
            and canonical is not None
            and not any(item in lowered for item in INVALID_META_RESPONSES)
        ):
            return str(last_value).strip(), canonical, False
        print(
            f'invalid raw {modality} explanation; retrying '
            f'({attempt + 1}/{max(args.max_generation_attempts, 1)}): '
            f'{last_value}'
        )
    fallback = empty_observation_description(modality)
    return str(last_value or '').strip(), fallback, True


@torch.inference_mode()
def generate(
    model,
    processor,
    modality,
    prompt,
    media_path,
    args,
    text_raw_few_shot=False,
    generation_profile=None,
):
    if text_raw_few_shot:
        if modality != 'text':
            raise ValueError('text_raw_few_shot is only valid for text input')
        return generate_from_messages(
            model,
            processor,
            [
                {'role': 'system', 'content': TEXT_RAW_SYSTEM},
                {'role': 'user', 'content': TEXT_RAW_EXAMPLE_USER},
                {'role': 'assistant', 'content': TEXT_RAW_EXAMPLE_ASSISTANT},
                {'role': 'user', 'content': prompt},
            ],
            modality,
            media_path,
            args,
            generation_profile=generation_profile,
        )
    if modality == 'vision':
        frames = extract_video_frames(media_path, args.num_video_frames)
        content = [
            {'type': 'image', 'image': frame}
            for frame in frames
        ]
        content.append({'type': 'text', 'text': prompt})
    elif modality == 'audio':
        # Match the supplied audio experiment: audio first, instruction second.
        content = [
            {'type': 'audio', 'audio': media_path},
            {'type': 'text', 'text': prompt},
        ]
    else:
        content = [{'type': 'text', 'text': prompt}]
    messages = [{'role': 'user', 'content': content}]

    return generate_from_messages(
        model,
        processor,
        messages,
        modality,
        media_path,
        args,
        generation_profile=generation_profile,
    )


def extract_video_frames(video_path, num_frames=4):
    """Uniformly sample PIL RGB frames, following gemma_video_exp.py."""
    try:
        import cv2
        import numpy as np
        from PIL import Image
    except ImportError as error:
        raise ImportError(
            'Video frame extraction requires opencv-python, numpy, and Pillow.'
        ) from error

    capture = cv2.VideoCapture(video_path)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1:
        capture.release()
        raise ValueError(f'Video contains no decodable frames: {video_path}')
    sample_count = min(max(int(num_frames), 1), total_frames)
    indices = np.linspace(0, total_frames - 1, sample_count, dtype=int)
    frames = []
    for frame_index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
    capture.release()
    if not frames:
        raise ValueError(f'Failed to decode sampled frames from: {video_path}')
    return frames


def generate_from_messages(
    model,
    processor,
    messages,
    modality,
    media_path,
    args,
    generation_profile=None,
):
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
        return_dict=True,
        return_tensors='pt',
    )

    inputs = inputs.to(model.device)
    if (
        modality == 'vision'
        and inputs.get('pixel_values') is None
        and inputs.get('pixel_values_videos') is None
    ):
        raise RuntimeError(
            f'Video frames were not loaded by the processor: {media_path}. '
            f'Processor output keys: {list(inputs.keys())}'
        )
    if modality == 'audio' and inputs.get('input_features') is None:
        raise RuntimeError(
            f'Audio was not loaded by the processor: {media_path}. '
            f'Processor output keys: {list(inputs.keys())}'
        )
    generation_kwargs = {
        'raw_text': {
            'do_sample': True, 'temperature': 0.7, 'top_p': 0.8, 'top_k': 20,
        },
        'raw_audio': {
            'do_sample': True, 'temperature': 1.0, 'top_p': 0.95, 'top_k': 64,
        },
        'raw_vision': {
            'do_sample': True, 'temperature': 0.7, 'top_p': 0.8, 'top_k': 20,
        },
    }.get(generation_profile)
    if generation_kwargs is None:
        raise ValueError(f'Unknown generation profile: {generation_profile}')
    if generation_profile.startswith('raw_') and not args.reference_sampling:
        generation_kwargs = {'do_sample': False}
    profile_max_tokens = {
        'raw_text': args.text_max_new_tokens,
        'raw_audio': args.audio_max_new_tokens,
        'raw_vision': args.vision_max_new_tokens,
    }[generation_profile]
    max_new_tokens = args.max_new_tokens or profile_max_tokens
    tokenizer = getattr(processor, 'tokenizer', None)
    if tokenizer is not None:
        generation_kwargs.update({
            'pad_token_id': tokenizer.pad_token_id,
            'eos_token_id': tokenizer.eos_token_id,
        })
    generation_kwargs['repetition_penalty'] = 1.0
    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        **generation_kwargs,
    )
    prompt_length = inputs['input_ids'].shape[-1]
    return processor.decode(
        generated[0, prompt_length:], skip_special_tokens=True
    ).strip()
