"""Versioned generation text. Default runtime stays on the citation prompt."""
from pathlib import Path

VERSIONS = ("pingo-v2-citations", "pingo-v1-reconstructed", "pingo-v3-verbatim-citations")
ROOT = Path(__file__).resolve().parents[2] / "prompts"


def load_prompt(version: str) -> str:
    if version not in VERSIONS:
        raise ValueError("Unknown prompt version")
    return (ROOT / f"{version}.txt").read_text()
