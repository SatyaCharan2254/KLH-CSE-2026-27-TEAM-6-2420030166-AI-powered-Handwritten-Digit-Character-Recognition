"""Local TrOCR handwriting recognition (no cloud API)."""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

logger = logging.getLogger(__name__)

MODEL_ID = "microsoft/trocr-small-handwritten"
ALLOWED_MIME_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
    "image/bmp",
    "image/gif",
}
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 20_000_000
MIN_SIDE = 8
MAX_LINES = 20
MIN_LINE_CONFIDENCE = 0.40
MIN_ACTIVE_ROW_RATIO = 0.08

_model_lock = threading.Lock()
_processor = None
_model = None
_device = None
_load_error: str | None = None


class OcrError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass
class LineResult:
    index: int
    text: str
    confidence: float = 0.0


@dataclass
class OcrResult:
    extracted_text: str
    processing_time_ms: int
    line_count: int
    lines: list[LineResult]
    warnings: list[str] = field(default_factory=list)
    model: str = MODEL_ID
    inverted: bool = False

    def to_api_dict(self) -> dict[str, Any]:
        payload = {
            "extractedText": self.extracted_text,
            "processingTimeMs": self.processing_time_ms,
            "lineCount": self.line_count,
            "lines": [
                {
                    "index": line.index,
                    "text": line.text,
                    "confidence": line.confidence,
                }
                for line in self.lines
            ],
            "warnings": self.warnings,
            "model": self.model,
        }
        return payload


def transformers_compatibility() -> dict[str, Any]:
    """Report whether the installed Transformers build can load TrOCR classes."""
    import importlib.metadata

    info: dict[str, Any] = {
        "modelId": MODEL_ID,
        "compatible": False,
        "transformersVersion": None,
        "torchVersion": None,
        "notes": [],
    }
    try:
        import torch

        info["torchVersion"] = torch.__version__
    except Exception as exc:
        info["notes"].append(f"PyTorch import failed: {exc.__class__.__name__}: {exc}")
        return info

    try:
        info["transformersVersion"] = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        info["notes"].append("The transformers package is not installed.")
        return info

    try:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel  # noqa: F401

        info["compatible"] = all(
            callable(getattr(model_class, "from_pretrained", None))
            for model_class in (TrOCRProcessor, VisionEncoderDecoderModel)
        )
        if info["compatible"]:
            info["notes"].append(
                "TrOCRProcessor and VisionEncoderDecoderModel imported successfully."
            )
        else:
            info["notes"].append("Required TrOCR classes do not expose from_pretrained.")
    except Exception as exc:
        info["notes"].append(
            f"TrOCR classes failed to import: {exc.__class__.__name__}: {exc}"
        )
    return info


def _get_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(force: bool = False):
    """Load TrOCR once and reuse it across requests."""
    global _processor, _model, _device, _load_error

    with _model_lock:
        if _model is not None and _processor is not None and not force:
            return _processor, _model, _device
        if _load_error and not force:
            raise OcrError(_load_error, status_code=500)

        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel

            _device = _get_device()
            logger.info("Loading %s on %s", MODEL_ID, _device)
            try:
                _processor = TrOCRProcessor.from_pretrained(MODEL_ID, use_fast=False)
            except TypeError:
                _processor = TrOCRProcessor.from_pretrained(MODEL_ID)

            _model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID)
            _model.to(_device)
            _model.eval()
            _load_error = None
            return _processor, _model, _device
        except Exception as exc:
            _processor = None
            _model = None
            _load_error = (
                "Failed to load microsoft/trocr-small-handwritten. "
                f"{exc.__class__.__name__}: {exc}"
            )
            logger.exception("TrOCR load failed")
            raise OcrError(_load_error, status_code=500) from exc


def model_loaded() -> bool:
    return _processor is not None and _model is not None


def decode_image(image_bytes: bytes, mime_type: str | None) -> Image.Image:
    if mime_type is not None and (
        not isinstance(mime_type, str) or mime_type.lower() not in ALLOWED_MIME_TYPES
    ):
        raise OcrError(
            f"Unsupported image type '{mime_type}'. Use PNG, JPEG, WEBP, BMP, or GIF."
        )
    if not image_bytes:
        raise OcrError("Image data is empty.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise OcrError(
            f"Image is too large ({len(image_bytes)} bytes). Maximum is {MAX_IMAGE_BYTES} bytes."
        )

    try:
        image = Image.open(BytesIO(image_bytes))
        image.load()
        image = ImageOps.exif_transpose(image)
    except Image.DecompressionBombError as exc:
        raise OcrError("Image resolution is too high. Please upload a smaller image.") from exc
    except UnidentifiedImageError as exc:
        raise OcrError("Invalid image data. The file could not be decoded.") from exc
    except OSError as exc:
        raise OcrError("Invalid or corrupted image data.") from exc

    if image.width < MIN_SIDE or image.height < MIN_SIDE:
        raise OcrError("Image is too small to recognize. Draw or upload a larger sample.")
    if image.width * image.height > MAX_PIXELS:
        raise OcrError("Image resolution is too high. Please upload a smaller image.")

    if image.mode in {"P", "RGBA", "LA"}:
        background = Image.new("RGB", image.size, (255, 255, 255))
        rgba = image.convert("RGBA")
        background.paste(rgba, mask=rgba.split()[-1])
        image = background
    else:
        image = image.convert("RGB")

    return image


def _should_invert(gray: np.ndarray) -> bool:
    """IAM-style TrOCR expects dark ink on a light background."""
    return float(gray.mean()) < 127.0


def prepare_document(image: Image.Image) -> tuple[Image.Image, np.ndarray, bool]:
    rgb = image.convert("RGB")
    gray = np.asarray(ImageOps.grayscale(rgb), dtype=np.uint8)
    inverted = _should_invert(gray)
    if inverted:
        rgb = ImageOps.invert(rgb)
        gray = 255 - gray
    return rgb, gray, inverted


def _ink_mask(gray: np.ndarray) -> np.ndarray:
    # Darker pixels are treated as ink after inversion to light background.
    threshold = min(200, max(90, int(gray.mean()) - 15))
    return gray < threshold


def split_line_boxes(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Return (left, top, right, bottom) boxes for text lines via row projection."""
    height, width = gray.shape
    ink = _ink_mask(gray)
    ink_ratio = float(ink.mean())
    if ink_ratio < 0.001:
        return []

    row_hist = ink.sum(axis=1).astype(np.float64)
    active = row_hist > max(3.0, row_hist.max() * MIN_ACTIVE_ROW_RATIO)

    ranges: list[tuple[int, int]] = []
    start = None
    for idx, is_active in enumerate(active):
        if is_active and start is None:
            start = idx
        elif not is_active and start is not None:
            if idx - start >= 6:
                ranges.append((start, idx))
            start = None
    if start is not None and height - start >= 6:
        ranges.append((start, height))

    if not ranges:
        return [(0, 0, width, height)]

    merged: list[tuple[int, int]] = []
    for top, bottom in ranges:
        if merged and top - merged[-1][1] <= 8:
            merged[-1] = (merged[-1][0], bottom)
        else:
            merged.append((top, bottom))

    boxes: list[tuple[int, int, int, int]] = []
    for top, bottom in merged:
        band = ink[top:bottom]
        col_hist = band.sum(axis=0)
        cols = np.where(col_hist > 0)[0]
        if len(cols) == 0:
            continue
        left = int(max(0, cols[0] - 8))
        right = int(min(width, cols[-1] + 9))
        pad_top = int(max(0, top - 6))
        pad_bottom = int(min(height, bottom + 6))
        boxes.append((left, pad_top, right, pad_bottom))
    return boxes or [(0, 0, width, height)]


def crop_lines(
    rgb: Image.Image, gray: np.ndarray, normalize_contrast: bool = True
) -> tuple[list[Image.Image], list[str]]:
    warnings: list[str] = []
    height, width = gray.shape
    boxes = split_line_boxes(gray)
    if not boxes:
        return [], ["No ink was detected in the image."]

    if len(boxes) > MAX_LINES:
        warnings.append(
            f"Detected {len(boxes)} line bands; only the first {MAX_LINES} were recognized."
        )
        boxes = boxes[:MAX_LINES]

    if len(boxes) > 1:
        warnings.append(
            "TrOCR is a single-line model. This image was split into line crops "
            "with a simple projection method, which can miss columns, diagrams, "
            "or overlapping handwriting."
        )
    else:
        left, top, right, bottom = boxes[0]
        if right - left >= width * 0.9 and height <= (bottom - top) * 1.8:
            return [rgb], warnings

    normalize_contrast = normalize_contrast and len(boxes) > 1
    crops = []
    for box in boxes:
        crop = rgb.crop(box)
        if crop.width < MIN_SIDE or crop.height < MIN_SIDE:
            continue
        if normalize_contrast:
            crop = ImageOps.autocontrast(ImageOps.grayscale(crop)).convert("RGB")
        crops.append(crop)

    if not crops:
        return [rgb], warnings
    return crops, warnings


def _recognize_line(
    processor, model, device, line_image: Image.Image
) -> tuple[str, float]:
    import torch

    pixel_values = processor(images=line_image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    with torch.no_grad():
        generated = model.generate(
            pixel_values,
            max_new_tokens=64,
            return_dict_in_generate=True,
            output_scores=True,
        )
    text = processor.batch_decode(generated.sequences, skip_special_tokens=True)[0]
    token_probabilities = []
    sequence = generated.sequences[0]
    for step, step_scores in enumerate(generated.scores):
        sequence_index = step + 1
        if sequence_index >= len(sequence):
            break
        token_id = int(sequence[sequence_index])
        if token_id == processor.tokenizer.eos_token_id:
            break
        token_probabilities.append(
            float(step_scores[0].softmax(-1)[token_id].item())
        )
    confidence = (
        math.exp(
            sum(math.log(max(probability, 1e-12)) for probability in token_probabilities)
            / len(token_probabilities)
        )
        if token_probabilities
        else 0.0
    )
    return (text or "").strip(), confidence


def recognize_image(image: Image.Image) -> OcrResult:
    started = time.perf_counter()
    rgb, gray, inverted = prepare_document(image)
    ink_ratio = float(_ink_mask(gray).mean())
    if ink_ratio < 0.001:
        raise OcrError(
            "No handwriting was detected. Draw darker strokes or upload a clearer image."
        )

    crops, warnings = crop_lines(rgb, gray, normalize_contrast=not inverted)
    processor, model, device = load_model()

    lines: list[LineResult] = []
    for index, crop in enumerate(crops):
        try:
            text, confidence = _recognize_line(processor, model, device, crop)
        except Exception as exc:
            logger.exception("TrOCR inference failed on line %s", index)
            raise OcrError(
                f"Recognition failed: {exc.__class__.__name__}. Check the image and try again.",
                status_code=500,
            ) from exc
        if text and confidence < MIN_LINE_CONFIDENCE:
            warnings.append(
                f"Line {index + 1} was omitted because recognition confidence was low."
            )
            text = ""
        elif not text:
            warnings.append(f"The model returned empty text for line {index + 1}.")
        lines.append(LineResult(index=index, text=text, confidence=confidence))

    extracted = "\n".join(line.text for line in lines if line.text).strip()

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return OcrResult(
        extracted_text=extracted,
        processing_time_ms=elapsed_ms,
        line_count=len(lines),
        lines=lines,
        warnings=warnings,
        inverted=inverted,
    )


def levenshtein(a, b) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            insert = current[j - 1] + 1
            delete = prev[j] + 1
            replace = prev[j - 1] + (ca != cb)
            current.append(min(insert, delete, replace))
        prev = current
    return prev[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    if not reference and not hypothesis:
        return 0.0
    if not reference:
        return 1.0
    return levenshtein(reference, hypothesis) / len(reference)


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words and not hyp_words:
        return 0.0
    if not ref_words:
        return 1.0
    return levenshtein(ref_words, hyp_words) / len(ref_words)
