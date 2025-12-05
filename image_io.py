"""Helpers for image input/output used by gemini3_image_tool."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Optional

from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "input"
MAX_OUTPUT_SUFFIX = 10


def resolve_image_path(image_path: str) -> Path:
    """Locate an input image, defaulting to ./input/ for bare filenames."""

    raw_path = Path(image_path)
    candidates = []

    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.append(raw_path)
        candidates.append(SCRIPT_DIR / raw_path)
        candidates.append(INPUT_DIR / raw_path)

    checked = []
    for candidate in candidates:
        if candidate in checked:
            continue
        checked.append(candidate)
        if candidate.exists():
            return candidate

    searched_locations = ", ".join(str(path) for path in checked)
    raise FileNotFoundError(
        f"Image '{image_path}' not found. Looked in: {searched_locations}"
    )


def _bytes_to_pillow_image(blob: bytes) -> Optional[Image.Image]:
    """
    Convert raw bytes into a Pillow Image and return None if decoding fails.
    """

    if not blob:
        return None
    try:
        return Image.open(io.BytesIO(blob))
    except Exception:
        return None


def _extract_blob_from_part(part) -> Optional[bytes]:
    """
    Best-effort extraction of raw binary data from a google.genai.parts.Part.
    """

    for attr_name in ("as_bytes", "as_png_bytes", "as_jpeg_bytes"):
        if hasattr(part, attr_name):
            try:
                blob = getattr(part, attr_name)()
            except Exception:
                blob = None
            if blob:
                return blob

    inline_data = getattr(part, "inline_data", None)
    if inline_data is not None:
        data = getattr(inline_data, "data", None)
        if data:
            if isinstance(data, str):
                try:
                    return base64.b64decode(data, validate=True)
                except Exception:
                    pass
            elif isinstance(data, (bytes, bytearray)):
                return bytes(data)

    file_data = getattr(part, "file_data", None)
    if file_data is not None:
        data = getattr(file_data, "data", None)
        if data:
            if isinstance(data, (bytes, bytearray)):
                return bytes(data)
            if isinstance(data, str):
                try:
                    return base64.b64decode(data, validate=True)
                except Exception:
                    pass
    return None


def _part_to_pillow_image(part) -> Optional[Image.Image]:
    """
    Convert a google.genai response part into a Pillow Image if possible.
    """

    for method_name in ("as_pil_image", "as_pillow_image", "as_image"):
        if hasattr(part, method_name):
            try:
                candidate = getattr(part, method_name)()
            except Exception:
                candidate = None
            if isinstance(candidate, Image.Image):
                return candidate
            if candidate is not None:
                for inner in ("as_pil_image", "as_pillow_image"):
                    if hasattr(candidate, inner):
                        try:
                            nested = getattr(candidate, inner)()
                        except Exception:
                            nested = None
                        if isinstance(nested, Image.Image):
                            return nested
    blob = _extract_blob_from_part(part)
    if blob:
        return _bytes_to_pillow_image(blob)
    return None


def _collect_response_parts(response) -> list:
    """
    Return a best-effort list of response parts that may contain images.
    """

    if response is None:
        return []

    def _parts_from_content(content) -> list:
        if not content:
            return []
        parts = getattr(content, "parts", None)
        return list(parts) if parts else []

    def _ensure_iterable(value):
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    direct_parts = getattr(response, "parts", None)
    if direct_parts:
        return list(direct_parts)

    collected = []
    for content in _ensure_iterable(getattr(response, "contents", None)):
        collected.extend(_parts_from_content(content))
    if collected:
        return collected

    for candidate in _ensure_iterable(getattr(response, "candidates", None)):
        collected.extend(_parts_from_content(getattr(candidate, "content", None)))
    if collected:
        return collected

    for content in _ensure_iterable(getattr(response, "output", None)):
        collected.extend(_parts_from_content(content))
    if collected:
        return collected

    for content in _ensure_iterable(getattr(response, "outputs", None)):
        collected.extend(_parts_from_content(content))
    return collected


def _next_available_output_filename(output_dir: Path, prefix: str):
    """Return a filename with suffix 0..MAX_OUTPUT_SUFFIX that does not yet exist."""

    for suffix in range(MAX_OUTPUT_SUFFIX + 1):
        candidate = output_dir / f"{prefix}_{suffix}.png"
        if not candidate.exists():
            return candidate, suffix
    return None


def save_images_from_response(
    response,
    output_prefix: str,
    expected_long_side: Optional[int] = None,
) -> int:
    """
    Extract image parts from the response and save them as PNG.
    """

    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for part in _collect_response_parts(response):
        image = _part_to_pillow_image(part)
        if image is not None:
            slot = _next_available_output_filename(output_dir, output_prefix)
            if not slot:
                print(
                    "No available output filenames. "
                    f"Existing files cover {output_prefix}_0.png through "
                    f"{output_prefix}_{MAX_OUTPUT_SUFFIX}.png in {output_dir}."
                )
                break
            filename, suffix = slot
            image.save(str(filename))
            width, height = image.size
            long_side = max(width, height)
            print(f"Saved image {suffix} ({width}x{height}) → {filename}")
            if expected_long_side and long_side < expected_long_side:
                print(
                    "WARNING: Model returned a smaller long side "
                    f"({long_side}px) than requested ({expected_long_side}px)."
                )
            saved += 1

    if saved == 0:
        print(
            "No image parts found in response "
            "(request may have been blocked by safety filters or returned only text)."
        )
        prompt_feedback = getattr(response, "prompt_feedback", None)
        block_reason = getattr(prompt_feedback, "block_reason", None)
        if block_reason:
            print(f"Prompt feedback block reason: {block_reason}")
        safety_ratings = getattr(prompt_feedback, "safety_ratings", None) or []
        if safety_ratings:
            for rating in safety_ratings:
                category = getattr(rating, "category", None)
                probability = getattr(rating, "probability", None)
                if category or probability:
                    print(
                        f"- Safety rating: category={category}, probability={probability}"
                    )
    return saved
