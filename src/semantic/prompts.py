"""Original v11 prompt text and generation defaults, byte-for-byte values."""

import os

PROMPT_VERSION = 'reference_single_pass_rule_clean_v11'

MODALITY_PROMPT_VERSIONS = {
    'text': 'text_summary_single_pass_v10',
    'audio': 'audio_acoustic_evidence_single_pass_v11',
    'vision': 'video_explanation_single_pass_v10',
}

TEXT_RAW_SYSTEM = (
    'You are an expert in sentiment analysis. The first user-assistant pair below '
    'is an EXAMPLE of the task. The second user message contains the ACTUAL text '
    "to analyze. For the actual text, output exactly one sentence in the format "
    "'Summary: <your analysis>'. Do NOT include any extra text or explanation."
)

TEXT_RAW_EXAMPLE_USER = (
    'Text: but to any of you who are i expect a thumbs up.'
)

TEXT_RAW_EXAMPLE_ASSISTANT = (
    'Summary: The statement expresses confidence in a shared understanding or '
    'experience.'
)

TONE_LABELS = (
    'neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'surprise', 'disgust',
)

TONE_ALIASES = {
    'fear': 'fearful',
    'frightened': 'fearful',
    'scared': 'fearful',
    'surprised': 'surprise',
    'surprising': 'surprise',
    'disgusted': 'disgust',
    'anger': 'angry',
    'happiness': 'happy',
    'joyful': 'happy',
    'joy': 'happy',
    'enthusiastic': 'happy',
    'cheerful': 'happy',
    'sadness': 'sad',
}

RAW_PROMPTS = {
    'text': 'Text: {text}',
    'audio': (
        'You are an expert in speech-prosody and audio emotion analysis. Listen '
        'only to the supplied audio clip. First identify two to four of the most '
        'distinctive observable acoustic cues, choosing from speech rate, pitch '
        'level and movement, energy or loudness, pauses, emphasis, rhythm, and '
        'voice quality. Describe actual cues and changes over time; do not fill '
        'every category with generic values. Then infer one tone from: neutral, '
        'calm, happy, sad, angry, fearful, surprise, disgust. The inference must '
        'be grounded in the stated acoustic cues. If evidence is weak or '
        'conflicting, say so and use weak confidence rather than inventing cues. '
        'Return exactly two sentences in this format: "Acoustic evidence: '
        '<concise observable cues and dynamics>. Inferred tone: <one allowed '
        'tone>; confidence: <weak, moderate, or strong>." Do not transcribe '
        'speech, infer identity, mention a dataset, give a numeric score, or add '
        'reasoning, alternatives, headings, bullets, or markdown.'
    ),
    'vision': (
        'You are an expert in video understanding and visual analysis. Your task '
        'is to watch the provided video and generate a detailed, coherent '
        'description of its content, following a chronological structure and '
        'focusing on observable details. The supplied images are uniformly '
        'sampled chronological frames from one continuous video; analyze their '
        'progression directly and never discuss the fact that they are still '
        'images or claim that a video is missing. Requirements: 1. Chronological Order: '
        'Describe the video in time sequence, using phrases like "In the '
        'beginning...", "In the following scene...", "Then...", "Finally...". '
        '2. Specific Physical Cues: Focus on concrete, visible details rather than '
        'abstract interpretations: facial expressions (mouth shape, eyebrows, '
        'eyes, and gaze direction), gestures and body language (hand movements, '
        'posture changes, head tilts, thumbs-up), character appearance (gender, '
        'approximate age, hair, clothing) if relevant, and setting only when it '
        'helps explain the context. 3. Cautious Emotion Inference: When inferring '
        'emotions, use hedging language such as "may indicate", "suggests", "is '
        'often considered", or "could be interpreted as". Never state emotions '
        'as absolute facts. 4. One Coherent Paragraph: Output exactly one '
        'paragraph in natural English. Do not use bullet points, titles, prefixes, '
        'or Markdown. 5. Length: Aim for 3-5 sentences, but adjust if the video '
        'has more or fewer actions. 6. No External References: Rely solely on the '
        'visual content; do not mention labels, datasets, or background knowledge. '
        'Example style only: "The screen shows a male character. In the beginning, '
        'his facial expression is relatively calm, with a slight upward curve at '
        'the corners of his mouth, which may indicate a positive emotion. In the '
        'following scene, he gives a thumbs-up gesture with his right hand, which '
        'is often considered a sign of approval or agreement." Now analyze the '
        'given video and produce the description.'
    ),
}

INVALID_VISION_RESPONSES = (
    'please provide the video',
    "i don't see a video",
    'no video was provided',
    'there is no video provided',
    'cannot access the video',
    'unable to view the video',
)

INVALID_META_RESPONSES = (
    'here are a few options',
    'option 1',
    'desired nuance',
    'most balanced',
    'i can help',
    'thinking process',
    'analyze the request',
    'deconstruction & filtering',
    'deconstruction and filtering',
    '**constraints:**',
    'i cannot perform this task',
    'you have not provided an audio',
    'no audio clip',
    'four static images',
    'prompt implies watching a video',
    'i have four static images',
    'descriptions should be based on',
)

DEFAULT_MODEL_PATH = os.environ.get(
    'GEMMA_MODEL_PATH',
    'models/gemma-4-E4B-it',
)
