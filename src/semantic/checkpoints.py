"""Atomic JSON writes, periodic backups and recovery. One counter per process."""

import os
import json
import re
import shutil

_ATOMIC_WRITE_COUNT = 0
_BACKUP_INTERVAL = 25


def read_json_checkpoint(path, repair_wrapping=False):
    """Read JSON and optionally repair harmless byte/prefix corruption."""
    try:
        with open(path, 'rb') as file:
            raw = file.read()
    except OSError:
        return None
    try:
        text = raw.decode('utf-8-sig')
    except UnicodeDecodeError:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        if not repair_wrapping:
            return None

    # NUL padding and accidental terminal/log prefixes cause an error at line 1,
    # column 1 even when the complete JSON object is still present.
    cleaned = text.replace('\x00', '').strip()
    starts = [0]
    object_match = re.search(r'\{\s*"train"\s*:', cleaned)
    if object_match:
        starts.append(object_match.start())
    first_brace = cleaned.find('{')
    if first_brace >= 0:
        starts.append(first_brace)
    last_brace = cleaned.rfind('}')
    for start in dict.fromkeys(starts):
        candidate = cleaned[start:last_brace + 1]
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and all(
            split in parsed for split in ('train', 'valid', 'test')
        ):
            return parsed
    return None


def atomic_write_json(data, path):
    global _ATOMIC_WRITE_COUNT
    _ATOMIC_WRITE_COUNT += 1
    absolute_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(absolute_path), exist_ok=True)
    temporary_path = absolute_path + '.tmp'
    backup_path = absolute_path + '.bak'
    with open(temporary_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.flush()
        os.fsync(file.fileno())
    # Keep the last parseable checkpoint. A zero-length or partially written
    # destination must never overwrite a valid backup.
    should_backup = (
        _ATOMIC_WRITE_COUNT == 1
        or _ATOMIC_WRITE_COUNT % _BACKUP_INTERVAL == 0
    )
    if should_backup and read_json_checkpoint(absolute_path) is not None:
        shutil.copy2(absolute_path, backup_path)
    os.replace(temporary_path, absolute_path)
