import base64
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PROJECT.ocr import OcrResult, LineResult
from PROJECT.server import app


def png_data_url(size=(80, 40), color=(0, 0, 0), background=(255, 255, 255)):
    image = Image.new("RGB", size, background)
    for x in range(10, size[0] - 10):
        for y in range(size[1] // 2 - 3, size[1] // 2 + 3):
            image.putpixel((x, y), color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return "data:image/png;base64," + encoded


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_missing_payload(self):
        response = self.client.post("/run-ocr", json={})
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_non_object_payload(self):
        response = self.client.post("/run-ocr", json=[])
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], "Request body must be a JSON object.")

    def test_request_too_large_returns_json_error(self):
        response = self.client.post(
            "/run-ocr",
            data="x" * (12 * 1024 * 1024 + 1),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 413)
        self.assertIn("error", response.get_json())

    def test_non_json(self):
        response = self.client.post("/run-ocr", data="x", content_type="text/plain")
        self.assertEqual(response.status_code, 400)

    def test_rejects_mismatched_data_url_mime_type(self):
        response = self.client.post(
            "/run-ocr",
            json={
                "imageBase64": png_data_url(),
                "fileMimeType": "image/jpeg",
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("MIME type does not match", response.get_json()["error"])

    def test_rejects_non_string_mime_type(self):
        response = self.client.post(
            "/run-ocr",
            json={"imageBase64": "AAAA", "fileMimeType": ["image/png"]},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_invalid_base64_image(self):
        response = self.client.post(
            "/run-ocr",
            json={"imageBase64": "not-valid-base64!!!", "fileMimeType": "image/png"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], "Invalid image data.")

    def test_success_keeps_extracted_text(self):
        data_url = png_data_url()
        encoded_image = data_url.split(",", 1)[1]
        expected = Image.open(io.BytesIO(base64.b64decode(encoded_image))).convert("RGB")
        received = []

        fake = OcrResult(
            extracted_text="hello",
            processing_time_ms=12,
            line_count=1,
            lines=[LineResult(index=0, text="hello", confidence=0.93)],
            warnings=[],
        )

        def capture_image(image):
            received.append(image.copy())
            return fake

        with patch(
            "PROJECT.server.recognize_image",
            side_effect=capture_image,
        ):
            response = self.client.post(
                "/run-ocr",
                json={"imageBase64": data_url, "fileMimeType": "image/png"},
            )
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["extractedText"], "hello")
        self.assertEqual(body["processingTimeMs"], 12)
        self.assertEqual(body["lineCount"], 1)
        self.assertEqual(body["lines"][0]["confidence"], 0.93)
        self.assertEqual(received[0].size, expected.size)
        self.assertEqual(received[0].tobytes(), expected.tobytes())

    def test_score_endpoint(self):
        response = self.client.post(
            "/score", json={"reference": "cat", "hypothesis": "car"}
        )
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertAlmostEqual(body["cer"], round(1 / 3, 4))

    def test_score_requires_object_and_string_fields(self):
        non_object = self.client.post("/score", json=[])
        invalid_fields = self.client.post(
            "/score", json={"reference": 1, "hypothesis": "cat"}
        )
        self.assertEqual(non_object.status_code, 400)
        self.assertEqual(invalid_fields.status_code, 400)

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertTrue(body["ok"])
        self.assertIn("modelLoaded", body)
        self.assertEqual(body["model"], "microsoft/trocr-small-handwritten")


if __name__ == "__main__":
    unittest.main()
