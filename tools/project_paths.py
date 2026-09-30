"""Resolve saved pre-organisation artifact paths without rewriting experiment evidence."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def resolve_artifact(saved_path):
    path = Path(saved_path)
    if path.is_absolute():
        return path
    name = str(saved_path).replace('\\', '/')
    mapping = json.loads((ROOT / 'docs/path_relocations.json').read_text(encoding='utf-8-sig'))
    for old, new in sorted(mapping.items(), key=lambda item: -len(item[0])):
        if name == old or name.startswith(old + '/'):
            return ROOT / (new + name[len(old):])
    return ROOT / name
