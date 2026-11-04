import os
import sys
from base64 import b64decode
from binascii import Error as Base64DecodeError
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
PUBLIC_DIR = ROOT_DIR / "public"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from PROJECT.evaluate import evaluate_labelled_samples, score_pair  # noqa: E402
from PROJECT.ocr import (  # noqa: E402
    MODEL_ID,
    OcrError,
    decode_image,
    load_model,
    model_loaded,
    recognize_image,
    transformers_compatibility,
)

load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024
CORS(
    app,
    resources={
        r"/run-ocr": {"origins": os.getenv("CORS_ORIGINS", "*")},
        r"/health": {"origins": os.getenv("CORS_ORIGINS", "*")},
        r"/evaluate": {"origins": os.getenv("CORS_ORIGINS", "*")},
        r"/score": {"origins": os.getenv("CORS_ORIGINS", "*")},
    },
)


@app.errorhandler(RequestEntityTooLarge)
def request_too_large(error):
    app.logger.warning("Rejected oversized request: %s", error.description)
    return jsonify({"error": "Request is too large. The maximum image size is 8 MB."}), 413


@app.get("/")
def index():
    return send_from_directory(PUBLIC_DIR, "index.html")


@app.get("/health")
def health():
    info = transformers_compatibility()
    return jsonify(
        {
            "ok": True,
            "model": MODEL_ID,
            "modelLoaded": model_loaded(),
            "transformersCompatible": info["compatible"],
            "transformersVersion": info["transformersVersion"],
            "torchVersion": info["torchVersion"],
            "notes": info["notes"],
        }
    )


@app.post("/run-ocr")
def run_ocr_endpoint():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    image_data = data.get("imageBase64")
    mime_type = data.get("fileMimeType")

    if not isinstance(image_data, str) or not image_data.strip():
        return jsonify({"error": "Missing image data or MIME type in request."}), 400
    if not isinstance(mime_type, str) or not mime_type.strip():
        return jsonify({"error": "Missing image data or MIME type in request."}), 400

    mime_type = mime_type.strip().lower()
    if image_data.startswith("data:"):
        header, separator, payload = image_data.partition(",")
        if not separator or ";base64" not in header.lower():
            return jsonify({"error": "Invalid image data URL."}), 400
        data_url_mime_type = header[5:].split(";", 1)[0].lower()
        if data_url_mime_type != mime_type:
            return jsonify({"error": "Image data URL MIME type does not match fileMimeType."}), 400
    else:
        payload = image_data

    try:
        image_bytes = b64decode(payload, validate=True)
        image = decode_image(image_bytes, mime_type)
        result = recognize_image(image)
        return jsonify(result.to_api_dict())
    except (Base64DecodeError, ValueError):
        return jsonify({"error": "Invalid image data."}), 400
    except OcrError as exc:
        return jsonify({"error": exc.message}), exc.status_code
    except Exception:
        app.logger.exception("Unexpected OCR failure")
        return jsonify({"error": "Recognition failed unexpectedly. Try another image."}), 500


@app.post("/score")
def score_endpoint():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400
    reference = data.get("reference")
    hypothesis = data.get("hypothesis")
    if not isinstance(reference, str) or not isinstance(hypothesis, str):
        return jsonify({"error": "Provide reference and hypothesis strings."}), 400
    return jsonify(score_pair(reference, hypothesis))


@app.post("/evaluate")
def evaluate_endpoint():
    try:
        return jsonify(evaluate_labelled_samples())
    except OcrError as exc:
        return jsonify({"error": exc.message}), exc.status_code
    except Exception:
        app.logger.exception("Evaluation failed")
        return jsonify({"error": "Could not run labelled evaluation."}), 500


def main():
    try:
        load_model()
    except OcrError as exc:
        app.logger.error("Model was not preloaded: %s", exc.message)
    app.run(debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
