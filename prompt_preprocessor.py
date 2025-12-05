"""
Lightweight prompt templating helpers.

Supports {{PLACEHOLDER}} tokens inside prompt text with optional defaults:
    {{ANIME_REFERENCE|121814056_p0.jpg}}

Values are provided through CLI flags (parsed via parse_prompt_variables) and
are substituted via render_prompt_text / render_prompt_file.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable

_PLACEHOLDER_PATTERN = re.compile(
    r"\{\{\s*([A-Za-z0-9_]+)(?:\|([^{}]*))?\s*\}\}"
)


def parse_prompt_variables(raw_pairs: Iterable[str]) -> Dict[str, str]:
    """
    Turn ["KEY=value", ...] into a {KEY: value} dict.
    """
    variables: Dict[str, str] = {}
    for raw in raw_pairs:
        if "=" not in raw:
            raise ValueError(
                f"Invalid --prompt-var '{raw}'. Expected KEY=VALUE."
            )
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(
                f"Invalid --prompt-var '{raw}'. KEY cannot be empty."
            )
        variables[key] = value
    return variables


def render_prompt_text(text: str, variables: Dict[str, str]) -> str:
    """
    Replace {{PLACEHOLDER}} tokens in text using the provided mapping.

    Placeholders may declare a default using {{PLACEHOLDER|fallback}}.
    """

    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        default = match.group(2).strip() if match.group(2) else None
        if key in variables:
            return variables[key]
        if default is not None:
            return default
        raise KeyError(
            f"Missing value for placeholder '{key}'. "
            "Provide it via --prompt-var KEY=VALUE."
        )

    return _PLACEHOLDER_PATTERN.sub(_replace, text)


def render_prompt_file(path: Path, variables: Dict[str, str]) -> str:
    """
    Shortcut to render a template file.
    """
    text = Path(path).read_text(encoding="utf-8")
    return render_prompt_text(text, variables)


__all__ = [
    "parse_prompt_variables",
    "render_prompt_file",
    "render_prompt_text",
]
