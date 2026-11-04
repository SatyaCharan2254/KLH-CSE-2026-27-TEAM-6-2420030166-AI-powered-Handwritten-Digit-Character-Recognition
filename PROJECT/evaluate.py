"""Evaluate TrOCR against labelled line images. Does not invent accuracy."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from PROJECT.ocr import (  # noqa: E402
    MODEL_ID,
    character_error_rate,
    recognize_image,
    word_error_rate,
)

EVAL_DIR = BASE_DIR / "evaluation"
SAMPLES_DIR = EVAL_DIR / "samples"
LABELS_PATH = EVAL_DIR / "labels.json"


def _font(size: int = 42):
    candidates = [
        Path(r"C:\Windows\Fonts\segoesc.ttf"),
        Path(r"C:\Windows\Fonts\inkfree.ttf"),
        Path(r"C:\Windows\Fonts\comic.ttf"),
        Path(r"C:\Windows\Fonts\calibri.ttf"),
        Path(r"C:\Windows\Fonts\arial.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def generate_labelled_samples() -> list[dict]:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    font = _font(48)
    specs = [
        ("line_hello.png", "hello"),
        ("line_quick.png", "the quick brown fox"),
        ("line_digits.png", "item 42"),
        ("multiline_hello_world.png", "hello\nworld"),
    ]
    labels = []
    for filename, text in specs:
        lines = text.split("\n")
        width = 720
        line_height = 70
        height = 40 + line_height * len(lines)
        image = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        y = 16
        for line in lines:
            draw.text((24, y), line, fill=(20, 20, 20), font=font)
            y += line_height
        path = SAMPLES_DIR / filename
        image.save(path)
        labels.append({"file": filename, "text": text, "kind": "synthetic-font"})
    LABELS_PATH.write_text(json.dumps(labels, indent=2), encoding="utf-8")
    return labels


def load_labels() -> list[dict]:
    if not LABELS_PATH.exists() or not SAMPLES_DIR.exists():
        return generate_labelled_samples()
    return json.loads(LABELS_PATH.read_text(encoding="utf-8"))


def evaluate_labelled_samples() -> dict:
    labels = load_labels()
    rows = []
    cer_values = []
    wer_values = []
    for item in labels:
        image_path = SAMPLES_DIR / item["file"]
        if not image_path.exists():
            rows.append(
                {
                    "file": item["file"],
                    "reference": item["text"],
                    "hypothesis": "",
                    "cer": None,
                    "wer": None,
                    "error": "Sample image is missing.",
                }
            )
            continue
        image = Image.open(image_path).convert("RGB")
        result = recognize_image(image)
        reference = item["text"]
        hypothesis = result.extracted_text
        cer = character_error_rate(reference, hypothesis)
        wer = word_error_rate(reference, hypothesis)
        cer_values.append(cer)
        wer_values.append(wer)
        rows.append(
            {
                "file": item["file"],
                "reference": reference,
                "hypothesis": hypothesis,
                "cer": round(cer, 4),
                "wer": round(wer, 4),
                "processingTimeMs": result.processing_time_ms,
                "lineCount": result.line_count,
                "kind": item.get("kind", "labelled"),
                "warnings": result.warnings,
            }
        )

    summary = {
        "model": MODEL_ID,
        "sampleCount": len(rows),
        "scoredCount": len(cer_values),
        "meanCer": round(sum(cer_values) / len(cer_values), 4) if cer_values else None,
        "meanWer": round(sum(wer_values) / len(wer_values), 4) if wer_values else None,
        "disclaimer": (
            "These scores are computed only against the labelled files in evaluation/. "
            "They are not a published accuracy claim for TrOCR. Synthetic font images "
            "are weaker than real handwriting samples; add your own labelled photos "
            "to evaluation/samples and evaluation/labels.json to measure real performance."
        ),
        "results": rows,
    }
    return summary


def score_pair(reference: str, hypothesis: str) -> dict:
    return {
        "reference": reference,
        "hypothesis": hypothesis,
        "cer": round(character_error_rate(reference, hypothesis), 4),
        "wer": round(word_error_rate(reference, hypothesis), 4),
    }


def main() -> int:
    print("Generating or loading labelled samples...")
    load_labels()
    print(f"Running {MODEL_ID} on evaluation/samples ...")
    summary = evaluate_labelled_samples()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
