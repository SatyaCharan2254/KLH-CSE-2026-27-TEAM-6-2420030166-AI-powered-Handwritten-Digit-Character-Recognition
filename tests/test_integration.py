import os
import sys
import unittest
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PROJECT.ocr import recognize_image, transformers_compatibility


def _font():
    for name in ("arial.ttf", "calibri.ttf", "segoeui.ttf"):
        path = Path(r"C:\Windows\Fonts") / name
        if path.exists():
            return ImageFont.truetype(str(path), 40)
    return ImageFont.load_default()


@unittest.skipUnless(
    os.getenv("RUN_TROCR_INTEGRATION") == "1",
    "Set RUN_TROCR_INTEGRATION=1 to load microsoft/trocr-small-handwritten",
)
class TrocrIntegrationTests(unittest.TestCase):
    def test_transformers_can_import_trocr(self):
        info = transformers_compatibility()
        self.assertTrue(info["compatible"], msg=info["notes"])

    def test_recognizes_printed_line_image(self):
        image = Image.new("RGB", (400, 80), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        draw.text((20, 18), "hello", fill=(0, 0, 0), font=_font())
        result = recognize_image(image)
        self.assertIsInstance(result.extracted_text, str)
        self.assertGreater(result.processing_time_ms, 0)
        normalized = "".join(ch.lower() for ch in result.extracted_text if ch.isalpha())
        self.assertIn("h", normalized)
        self.assertGreater(result.lines[0].confidence, 0.40)

    def test_suppresses_low_confidence_non_text_camera_icon(self):
        image = Image.new("RGB", (495, 371), (0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((207, 147, 288, 214), radius=9, outline=(255, 255, 255), width=5)
        draw.rounded_rectangle((229, 140, 263, 151), radius=4, outline=(255, 255, 255), width=5)
        draw.ellipse((232, 169, 262, 199), fill=(255, 255, 255))
        draw.line((198, 225, 296, 136), fill=(255, 255, 255), width=5)

        result = recognize_image(image)

        self.assertEqual(result.extracted_text, "")
        self.assertTrue(any("confidence was low" in warning for warning in result.warnings))
        self.assertLess(result.lines[0].confidence, 0.40)


if __name__ == "__main__":
    unittest.main()
