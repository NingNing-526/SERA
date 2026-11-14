"""Rule cleanup and fallback text. No additional LLM refinement stage."""

import re
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


def clean_model_response(value):
    """Remove leaked Gemma reasoning/channel markers before validation."""
    text = str(value).replace('\\</tr>', '</tr>').strip()
    # Standard thinking blocks, when both delimiters are present.
    text = re.sub(
        r'<think>.*?</think>', ' ', text, flags=re.IGNORECASE | re.DOTALL
    )
    # Gemma-4 occasionally decodes an end-of-turn/channel marker as literal
    # </tr>, "thought", or "earlier", followed by its private reasoning.  Keep
    # only the answer that precedes the first such marker.
    marker = re.search(
        r'(?:</?tr>|</?think>|[.!?]thought\b|'
        r'(?:^|[\r\n])\s*thought\s*(?:[\r\n:]|$)|'
        r'\bthinking process\s*:|'
        r"here(?:'s| is) a thinking process\b|"
        r'\bthere is no video provided\b|'
        r'\bthere are no subsequent frames provided\b|'
        # Examples observed in MOSI include "There is no fourth image
        # provided" appended after an otherwise valid chronological answer.
        r'\bthere (?:is|are) no [^.!?]{0,80}?\b(?:image|frame)s? provided\b|'
        r'\bthe description will conclude\b|'
        r'\bplease provide the video you would like me to describe\b|'
        r'\bare you ready for the next task\b|'
        r'\bwould you like me to\b|'
        r'\blet me know if you\b)',
        text,
        flags=re.IGNORECASE,
    )
    if marker is not None:
        # For an attached channel marker such as ".thought", retain the period
        # terminating the valid answer and discard everything after it.
        marker_text = marker.group(0).lower()
        keep_terminal = marker_text.startswith(('.', '!', '?'))
        text = text[:marker.start() + int(keep_terminal)]
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(
        r'([.!?])(?:earlier|thought)\s*$',
        r'\1',
        text,
        flags=re.IGNORECASE,
    )
    return ' '.join(text.split()).strip()


def validate_and_normalize_description(value, modality, source_text=None):
    normalized = clean_model_response(value)
    lowered = normalized.lower()
    if not normalized or any(item in lowered for item in INVALID_META_RESPONSES):
        return None
    if normalized.startswith('```') or normalized.endswith('```'):
        return None

    if modality == 'text':
        match = re.search(r'(?i)\bsummary\s*:\s*(.+)', normalized)
        if match is None:
            return None
        summary = match.group(1).strip(' "\'')
        first_sentence = re.match(r'(.+?[.!?])', summary)
        if first_sentence is not None:
            summary = first_sentence.group(1).strip()
        if not summary or len(summary.split()) > 80:
            return None
        # The reference text script asks for one Summary sentence.  Do not
        # remove emotion/polarity words: they are the intended semantic clue.
        return f'Summary: {summary}'

    if modality == 'audio':
        evidence_match = re.fullmatch(
            r'(?is)acoustic evidence\s*:\s*(.+?[.!?])\s*'
            r'inferred tone\s*:\s*'
            r'(neutral|calm|happy|sad|angry|fearful|fear|'
            r'frightened|scared|surprise|surprised|disgust|disgusted|anger|'
            r'happiness|joyful|joy|enthusiastic|cheerful|sadness)\s*;\s*'
            r'confidence\s*:\s*(weak|moderate|strong)\s*[.!]?',
            normalized,
        )
        if evidence_match is None:
            return None
        evidence = ' '.join(evidence_match.group(1).split()).strip()
        if len(evidence.split()) < 4 or len(evidence.split()) > 60:
            return None
        tone = evidence_match.group(2).lower()
        tone = TONE_ALIASES.get(tone, tone)
        if tone not in TONE_LABELS:
            return None
        confidence = evidence_match.group(3).lower()
        return (
            f'Acoustic evidence: {evidence} Inferred tone: {tone}; '
            f'confidence: {confidence}.'
        )

    if any(item in lowered for item in INVALID_VISION_RESPONSES):
        return None
    if lowered.startswith(
        ('no reliable chronological video', 'no chronological video explanation')
    ):
        return empty_observation_description('vision')
    if len(normalized.split()) > 180:
        return None
    # The video reference emits a natural paragraph, not a JSON concept table.
    # Accept concise descriptions for unusually short/static MOSI segments even
    # when the model cannot honestly produce all requested 3--5 sentences.
    complete_sentences = re.findall(r'[^.!?]+[.!?]', normalized)
    if not complete_sentences:
        return None
    normalized = ' '.join(
        sentence.strip() for sentence in complete_sentences[:5]
    )
    if re.search(r'[\u0400-\u04ff\u0e00-\u0e7f]', normalized):
        return None
    return normalized


def empty_observation_description(modality):
    if modality == 'text':
        return 'Summary: No reliable sentiment summary was generated.'
    if modality == 'audio':
        return (
            'Acoustic evidence: No reliable acoustic evidence was generated. '
            'Inferred tone: neutral; confidence: weak.'
        )
    return 'No reliable chronological video description was generated.'
