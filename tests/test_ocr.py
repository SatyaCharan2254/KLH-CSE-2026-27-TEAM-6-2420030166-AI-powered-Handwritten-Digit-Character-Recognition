import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PROJECT.ocr import (
    OcrError,
    character_error_rate,
    decode_image,
    crop_lines,
    prepare_document,
    recognize_image,
    split_line_boxes,
    transformers_compatibility,
    word_error_rate,
)


def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class DecodeTests(unittest.TestCase):
    def test_rejects_non_image_mime(self):
        with self.assertRaises(OcrError):
            decode_image(b"not-an-image", "application/pdf")

    def test_rejects_empty_bytes(self):
        with self.assertRaises(OcrError):
            decode_image(b"", "image/png")

    def test_rejects_undecodable_bytes(self):
        with self.assertRaises(OcrError):
            decode_image(b"this-is-not-png", "image/png")

    def test_accepts_png(self):
        image = Image.new("RGB", (40, 40), (255, 255, 255))
        decoded = decode_image(png_bytes(image), "image/png")
        self.assertEqual(decoded.size, (40, 40))
        self.assertEqual(decoded.mode, "RGB")

    def test_corrects_jpeg_exif_orientation(self):
        image = Image.new("RGB", (80, 40), (255, 255, 255))
        exif = Image.Exif()
        exif[274] = 6
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", exif=exif)

        decoded = decode_image(buffer.getvalue(), "image/jpeg")

        self.assertEqual(decoded.size, (40, 80))

    def test_rejects_tiny_image(self):
        image = Image.new("RGB", (4, 4), (255, 255, 255))
        with self.assertRaises(OcrError):
            decode_image(png_bytes(image), "image/png")


class LineSplitTests(unittest.TestCase):
    def test_detects_two_lines(self):
        image = Image.new("RGB", (200, 120), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        draw.rectangle((10, 10, 160, 28), fill=(0, 0, 0))
        draw.rectangle((10, 70, 160, 88), fill=(0, 0, 0))
        _, gray, inverted = prepare_document(image)
        self.assertFalse(inverted)
        boxes = split_line_boxes(gray)
        self.assertGreaterEqual(len(boxes), 2)

    def test_sparse_rows_do_not_merge_separate_text_lines(self):
        image = Image.new("RGB", (200, 100), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        draw.rectangle((10, 10, 160, 28), fill=(0, 0, 0))
        draw.rectangle((10, 70, 160, 88), fill=(0, 0, 0))
        for y in range(31, 69):
            for x in range(8):
                draw.point((x, y), fill=(0, 0, 0))

        _, gray, _ = prepare_document(image)
        boxes = split_line_boxes(gray)
        self.assertEqual(len(boxes), 2)

    def test_single_line_recognition_crops_surrounding_whitespace(self):
        image = Image.new("RGB", (200, 80), (255, 255, 255))
        ImageDraw.Draw(image).rectangle((20, 25, 100, 45), fill=(0, 0, 0))
        rgb, gray, _ = prepare_document(image)

        crops, _ = crop_lines(rgb, gray)

        self.assertEqual(len(crops), 1)
        self.assertLess(crops[0].width, image.width)

    def test_wide_single_line_preserves_the_original_line_image(self):
        image = Image.new("RGB", (400, 128), (255, 255, 255))
        ImageDraw.Draw(image).rectangle((0, 25, 399, 84), fill=(0, 0, 0))
        rgb, gray, _ = prepare_document(image)

        crops, _ = crop_lines(rgb, gray)

        self.assertEqual(len(crops), 1)
        self.assertEqual(crops[0].size, image.size)

    def test_inverts_white_ink_on_black(self):
        image = Image.new("RGB", (120, 60), (0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.rectangle((10, 20, 100, 36), fill=(255, 255, 255))
        _, gray, inverted = prepare_document(image)
        self.assertTrue(inverted)
        self.assertGreater(float(gray.mean()), 127)


class MetricTests(unittest.TestCase):
    def test_cer_exact(self):
        self.assertEqual(character_error_rate("hello", "hello"), 0.0)

    def test_cer_one_substitution(self):
        self.assertAlmostEqual(character_error_rate("cat", "car"), 1 / 3)

    def test_wer_word_swap(self):
        self.assertEqual(word_error_rate("hello world", "hello there"), 0.5)


class ConfidenceFilterTests(unittest.TestCase):
    def test_omits_low_confidence_line_instead_of_returning_guess(self):
        image = Image.new("RGB", (200, 80), (255, 255, 255))
        ImageDraw.Draw(image).line((20, 40, 180, 40), fill=(0, 0, 0), width=6)

        with (
            patch("PROJECT.ocr.load_model", return_value=(object(), object(), None)),
            patch("PROJECT.ocr._recognize_line", return_value=("unrelated", 0.1)),
        ):
            result = recognize_image(image)

        self.assertEqual(result.extracted_text, "")
        self.assertEqual(result.lines[0].text, "")
        self.assertEqual(result.lines[0].confidence, 0.1)
        self.assertTrue(any("confidence was low" in warning for warning in result.warnings))


class CompatibilityTests(unittest.TestCase):
    def test_compatibility_report_has_versions(self):
        info = transformers_compatibility()
        self.assertIn("compatible", info)
        self.assertIn("modelId", info)
        self.assertEqual(info["modelId"], "microsoft/trocr-small-handwritten")


if __name__ == "__main__":
    unittest.main()
