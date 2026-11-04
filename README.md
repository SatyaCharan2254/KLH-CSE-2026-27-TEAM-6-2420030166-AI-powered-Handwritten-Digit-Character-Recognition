# Handwritten Text Recognition

A Flask web app for recognizing handwritten text using Microsoft's TrOCR model
(`microsoft/trocr-small-handwritten`). Use the browser to write on a canvas, upload
an image, or capture one with a camera. Recognition runs in the Python backend;
the browser does not call a third-party OCR API.

TrOCR is a line-recognition model. The backend detects and crops text lines from
simple multi-line images, but columns, forms, and overlapping handwriting are not
reliably supported.

## Requirements

- Windows 10/11 or a compatible Python host
- Python 3.12 recommended for a fresh environment. The current code and real-model
  integration were also tested here with Python 3.14.2, CPU PyTorch 2.14.1, and
  Transformers 4.57.6.
- Internet access the first time the model is loaded, unless the model is already
  cached. Hugging Face model files are downloaded into the normal local cache.

## Install and run on Windows

Run these commands in PowerShell from the repository root:

```powershell
cd D:\AIML
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m PROJECT.server
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000). The first model load can
take a minute or more. If Python 3.12 is not installed, use `python` instead of
`py -3.12`; the existing environment was verified with Python 3.14.2.

The optional CORS setting is in `PROJECT/.env.example`. Copy it to
`PROJECT/.env` only if you need to configure cross-origin requests. Keep the
default local frontend and API on the same origin.

## Application flow

1. `public/index.html` captures canvas strokes, uploads, or camera frames and
   sends a base64 image to the Flask API.
2. `PROJECT/server.py` validates the JSON request, MIME type, and encoded image
   size, then returns a consistent JSON result or error.
3. `PROJECT/ocr.py` decodes and normalizes the image, separates simple line
   bands, loads the TrOCR processor/model once, and runs inference on CPU or CUDA
   when available.
4. The UI displays the recognized text, timing, line count, and any limitations
   reported by preprocessing.
5. `PROJECT/evaluate.py` scores the labelled files in `evaluation/` using CER
   and WER.

## API

`POST /run-ocr` accepts a JSON object:

```json
{
  "imageBase64": "data:image/png;base64,...",
  "fileMimeType": "image/png"
}
```

Successful responses include `extractedText`, `processingTimeMs`, `lineCount`,
`lines`, `warnings`, and `model`. Errors use `{"error": "..."}` with an
appropriate 4xx or 5xx status. The API accepts PNG, JPEG, WEBP, BMP, and GIF
images up to 8 MB (20 megapixels maximum); the JSON request limit allows for
base64 encoding overhead.

Each recognized line includes a model-token confidence score. Lines below the
conservative 0.40 cutoff are omitted from `extractedText` and reported in
`warnings` rather than presented as certain text. This score is a rejection
heuristic, not a calibrated probability of correctness.

Other routes:

- `GET /health` reports runtime compatibility and whether the model is loaded.
- `POST /score` accepts `{"reference": "...", "hypothesis": "..."}` and returns
  character and word error rates.
- `POST /evaluate` runs the bundled labelled evaluation set.

## Tests and evaluation

Run the unit/API tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Run tests that load the actual TrOCR model (the model must be cached or
downloadable):

```powershell
$env:RUN_TROCR_INTEGRATION = "1"
.\.venv\Scripts\python.exe -m unittest tests.test_integration -v
```

Measure the labelled samples:

```powershell
.\.venv\Scripts\python.exe -m PROJECT.evaluate
```

The bundled four samples are synthetic-font images, not real handwriting. The
last measured run on this environment recognized all four labels exactly (mean
CER 0.0000, mean WER 0.0000). This is only a score on those synthetic examples,
not a general handwriting accuracy claim. To measure your use case, add labelled
PNG/JPEG files to `evaluation/samples/` and entries to `evaluation/labels.json`.

The optional IAM-line streaming evaluation measured the first 100 real
handwriting samples with mean CER 0.0651, mean WER 0.1674, and 35% exact-match
accuracy after confidence filtering. Its predictions and report are in
`reports/ocr-fix-evaluation/`. To reproduce this network-backed evaluation,
install its optional streaming dependencies and run:

```powershell
.\.venv\Scripts\python.exe -m pip install fsspec pyarrow
.\.venv\Scripts\python.exe -m PROJECT.stream_evaluate --limit 100 --output-dir reports\ocr-fix-evaluation
```

This samples the `test` split of `Teklia/IAM-line`; it does not download the
entire dataset. Scores are specific to those 100 samples and this CPU run.

## Deployment

The supported, verified setup is the local Flask application above. The
repository contains a Vercel adapter/configuration, but the complete OCR backend
depends on PyTorch and downloads/loads a model at runtime; that serverless
deployment has not been validated and must not be treated as a working
production deployment. The existing Vercel project-root configuration also
needs to target the repository root for the root-level `api/`, `public/`, and
`evaluation/` directories to be present.

For hosted use, run the Flask application on a persistent Python host that
supports PyTorch, model downloads/cache, and the memory required for inference.
Use a production WSGI server supplied by that host rather than Flask's built-in
development server, and enable HTTPS for camera capture. No credentials or
third-party OCR API keys are required.

## Limitations

- Recognition quality depends on handwriting, image clarity, and line layout.
- Multi-line pages use simple horizontal projections; complex layouts are not
  fully supported.
- Notebook ruling lines are not explicitly removed and can still interfere with
  recognition.
- Camera access requires browser permission and a secure context (localhost or
  HTTPS).
- The bundled score is synthetic-font-only; real handwriting quality remains
  unmeasured until real labelled samples are added.
