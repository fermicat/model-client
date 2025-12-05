#!/usr/bin/env python3
"""
CLI helper for Gemini 3 Pro Image Preview (gemini-3-pro-image-preview).

Supports:
- Text → image generation
- Image(s) + text → edited image(s)

Usage examples:

  # Text → image
  python gemini3_image_tool.py generate \
    -p "3D render of a cute robot reading a book, studio lighting" \
    --aspect-ratio 16:9 --image-size 2K \
    --safety strict -o robot_book

  # Text → image using a prompt saved in ./prompt/cosplay.md
  python gemini3_image_tool.py generate \
    --prompt-file cosplay.md \
    --image-size 4K --safety relaxed -o cosplay_scene

  # Edit existing photo(s)
  python gemini3_image_tool.py edit \
    -p "Turn this into a watercolor painting at sunset." \
    --image my_photo.jpg \
    --image-size 1K \
    --safety relaxed -o my_photo_watercolor
"""

import argparse
from pathlib import Path
from typing import List, Optional

from google import genai
from google.genai import types
from PIL import Image
from dotenv import load_dotenv  # load .env
from image_io import resolve_image_path, save_images_from_response
from prompt_preprocessor import (
    parse_prompt_variables,
    render_prompt_file,
    render_prompt_text,
)

# Load variables from .env if present (GOOGLE_API_KEY / GEMINI_API_KEY, etc.)
load_dotenv()

SCRIPT_DIR = Path(__file__).resolve().parent
PROMPT_DIR = SCRIPT_DIR / "prompt"

DEFAULT_MODEL = "gemini-3-pro-image-preview"
DEFAULT_TEMPERATURE = 1.0

SUPPORTED_ASPECT_RATIOS = [
    "1:1", "2:3", "3:2", "3:4", "4:3",
    "4:5", "5:4", "9:16", "16:9", "21:9",
]
SUPPORTED_IMAGE_SIZES = ["1K", "2K", "4K"]  # ≈1024/2048/4096 px on long side
IMAGE_SIZE_TO_LONG_EDGE = {
    "1K": 1024,
    "2K": 2048,
    "4K": 4096,
}


def resolve_prompt_file(prompt_file: str) -> Path:
    """Locate a prompt file.

    Accepts either an absolute/relative path or a filename inside ./prompt.
    """

    raw_path = Path(prompt_file)
    candidates: List[Path] = []

    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.append(raw_path)
        candidates.append(SCRIPT_DIR / raw_path)
        candidates.append(PROMPT_DIR / raw_path)

        if raw_path.suffix == "":
            candidates.append((PROMPT_DIR / raw_path).with_suffix(".md"))

    checked = []
    for candidate in candidates:
        if candidate in checked:
            continue
        checked.append(candidate)
        if candidate.exists():
            return candidate

    searched_locations = ", ".join(str(path) for path in checked)
    raise FileNotFoundError(
        f"Prompt file '{prompt_file}' not found. Looked in: {searched_locations}"
    )


def get_prompt_text(
    prompt: Optional[str],
    prompt_file: Optional[str],
    prompt_vars: Optional[List[str]] = None,
) -> str:
    """Return the text prompt from inline input or a prompt file."""
    variables = parse_prompt_variables(prompt_vars or [])

    if prompt:
        return render_prompt_text(prompt, variables)

    if prompt_file:
        path = resolve_prompt_file(prompt_file)
        return render_prompt_file(path, variables)

    raise ValueError("Either prompt text or prompt file must be provided.")


def make_client(api_key: Optional[str] = None) -> genai.Client:
    """
    Create a Gemini client.

    - If api_key is provided, use that.
    - Otherwise, the SDK will look at GOOGLE_API_KEY / GEMINI_API_KEY env vars.
      load_dotenv() above ensures values from .env are visible here.
    """
    if api_key:
        return genai.Client(api_key=api_key)
    return genai.Client()


def build_image_config(aspect_ratio: Optional[str], image_size: Optional[str]):
    """
    Build an ImageConfig with aspect ratio and/or resolution.

    Both fields are optional; if nothing is set, we return None and let the model decide.
    """
    if not aspect_ratio and not image_size:
        return None

    config = types.ImageConfig()
    if aspect_ratio:
        if hasattr(config, "aspect_ratio"):
            config.aspect_ratio = aspect_ratio
        else:
            raise AttributeError(
                "Installed google-genai SDK does not expose aspect_ratio on ImageConfig."
            )

    if image_size:
        if hasattr(config, "image_size"):
            config.image_size = image_size  # older SDK builds
        elif hasattr(config, "size"):
            config.size = image_size  # newer SDK builds
        else:
            raise AttributeError(
                "Installed google-genai SDK does not expose the image size field."
            )

    return config
def build_safety_settings(preset: str = "relaxed"):
    """
    Build safety settings from a simple preset.

    Gemini models support these categories:
    - HARM_CATEGORY_HARASSMENT
    - HARM_CATEGORY_HATE_SPEECH
    - HARM_CATEGORY_SEXUALLY_EXPLICIT
    - HARM_CATEGORY_DANGEROUS_CONTENT
    """

    if preset == "default":
        # Let the API defaults apply (usually block medium+).
        return None

    categories = [
        types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
    ]

    if preset == "strict":
        # Block low / medium / high probability unsafe content
        threshold = types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
    elif preset == "relaxed":
        # Block only high probability unsafe content
        threshold = types.HarmBlockThreshold.BLOCK_ONLY_HIGH
    elif preset == "off":
        # Effectively disable blocking for these categories
        threshold = types.HarmBlockThreshold.BLOCK_NONE
    else:
        raise ValueError(f"Unknown safety preset: {preset}")

    return [
        types.SafetySetting(
            category=cat,
            threshold=threshold,
        )
        for cat in categories
    ]



def _expected_long_side(image_size: Optional[str]) -> Optional[int]:
    """
    Return the requested long-side resolution (in px) for a CLI --image-size flag.
    """
    if not image_size:
        return None
    return IMAGE_SIZE_TO_LONG_EDGE.get(image_size.upper())


def run_generate(client, args):
    """
    Text → image.
    """
    safety_settings = build_safety_settings(args.safety)
    image_config = build_image_config(args.aspect_ratio, args.image_size)
    response_modalities = ["IMAGE", "TEXT"] if args.include_text else ["IMAGE"]
    prompt_text = get_prompt_text(args.prompt, args.prompt_file, args.prompt_vars)

    config = types.GenerateContentConfig(
        response_modalities=response_modalities,
        image_config=image_config,
        temperature=DEFAULT_TEMPERATURE,
        top_k=args.top_k,
        seed=args.seed,
        safety_settings=safety_settings,
    )

    response = client.models.generate_content(
        model=args.model,
        contents=prompt_text,
        config=config,
    )

    expected_long_side = _expected_long_side(args.image_size)
    save_images_from_response(response, args.output, expected_long_side)

    if args.include_text:
        print("\n=== Model text description ===")
        print(response.text)


def run_edit(client, args):
    """
    Image(s) + text prompt → edited image(s).
    """
    safety_settings = build_safety_settings(args.safety)
    image_config = build_image_config(args.aspect_ratio, args.image_size)
    response_modalities = ["IMAGE", "TEXT"] if args.include_text else ["IMAGE"]
    prompt_text = get_prompt_text(args.prompt, args.prompt_file, args.prompt_vars)

    config = types.GenerateContentConfig(
        response_modalities=response_modalities,
        image_config=image_config,
        temperature=DEFAULT_TEMPERATURE,
        top_k=args.top_k,
        seed=args.seed,
        safety_settings=safety_settings,
    )

    images: List[Image.Image] = [Image.open(resolve_image_path(path)) for path in args.image]
    contents: List = [prompt_text] + images  # prompt first, then images

    response = client.models.generate_content(
        model=args.model,
        contents=contents,
        config=config,
    )

    expected_long_side = _expected_long_side(args.image_size)
    save_images_from_response(response, args.output, expected_long_side)

    if args.include_text:
        print("\n=== Model text description ===")
        print(response.text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Use Gemini 3 Pro Image Preview to generate or edit images."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Common args helper: added to BOTH subcommands so they work after the subcommand.
    def add_common_arguments(p: argparse.ArgumentParser):
        p.add_argument(
            "--model",
            default=DEFAULT_MODEL,
            help=f"Model name (default: {DEFAULT_MODEL})",
        )
        p.add_argument(
            "--aspect-ratio",
            choices=SUPPORTED_ASPECT_RATIOS,
            help=(
                "Aspect ratio like 1:1, 16:9, 9:16, 4:3, 3:2, 2:3, 4:5, 5:4, 21:9. "
                "If omitted, the model chooses automatically."
            ),
        )
        p.add_argument(
            "--image-size",
            choices=SUPPORTED_IMAGE_SIZES,
            help="Resolution: 1K≈1024px, 2K≈2048px, 4K≈4096px on the long side.",
        )
        p.add_argument(
            "--safety",
            choices=["default", "strict", "relaxed", "off"],
            default="relaxed",
            help=(
                "Safety preset. 'default' = API defaults; "
                "'strict' = block low & above; 'relaxed' = block only high; "
                "'off' = BLOCK_NONE (still subject to Google's hard safety limits)."
            ),
        )
        p.add_argument(
            "--top-k",
            type=int,
            default=32,
            help="Top-k sampling. Set to 0 to disable.",
        )
        p.add_argument(
            "--seed",
            type=int,
            default=None,
            help="Random seed for more repeatable outputs (optional).",
        )
        p.add_argument(
            "--include-text",
            action="store_true",
            help="Also return a text description along with the image.",
        )
        p.add_argument(
            "-o",
            "--output",
            default="output",
            help="Output filename prefix (default: output).",
        )

    def add_prompt_arguments(p: argparse.ArgumentParser, inline_help: str):
        prompt_group = p.add_mutually_exclusive_group(required=True)
        prompt_group.add_argument(
            "-p",
            "--prompt",
            help=inline_help,
        )
        prompt_group.add_argument(
            "--prompt-file",
            help=(
                "Use the contents of a prompt file instead of inline text. "
                "Provide either a path or the filename located under ./prompt/."
            ),
        )
        p.add_argument(
            "--prompt-var",
            dest="prompt_vars",
            action="append",
            default=[],
            metavar="KEY=VALUE",
            help=(
                "Template variable for prompt preprocessing. "
                "Repeat for multiple variables."
            ),
        )

    # --- generate subcommand ---
    gen_parser = subparsers.add_parser("generate", help="Text → image generation")
    add_common_arguments(gen_parser)
    add_prompt_arguments(
        gen_parser,
        inline_help="Text prompt describing the image you want.",
    )

    # --- edit subcommand ---
    edit_parser = subparsers.add_parser("edit", help="Edit existing image(s) with a prompt")
    add_common_arguments(edit_parser)
    add_prompt_arguments(
        edit_parser,
        inline_help="Edit instructions, e.g. 'Make this a watercolor painting at sunset'.",
    )
    edit_parser.add_argument(
        "-i",
        "--image",
        action="append",
        required=True,
        help=(
            "Path to an input image. Bare filenames are looked up under ./input/. "
            "You can pass multiple --image flags."
        ),
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    client = make_client()

    if args.command == "generate":
        run_generate(client, args)
    elif args.command == "edit":
        run_edit(client, args)
    else:
        parser.error("You must choose a subcommand: generate or edit")


if __name__ == "__main__":
    main()
