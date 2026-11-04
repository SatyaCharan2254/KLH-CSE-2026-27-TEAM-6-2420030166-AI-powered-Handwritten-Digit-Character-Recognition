# Project cleanup report

## Removed obsolete or duplicate files

| Path | Reason |
|---|---|
| `PROJECT/app.py` | One-line obsolete placeholder; local execution uses `python -m PROJECT.server`, and the Vercel adapter is `api/index.py`. |
| `PROJECT/public/index.html` | One-line placeholder; Flask serves the actual frontend from root `public/index.html`. |
| `PROJECT/vercel.json` | One-line obsolete placeholder; the sole checked-in Vercel routing configuration is root `vercel.json`. |
| `PROJECT/api.txt` | Unreferenced note duplicated by the README; removed its obsolete ignore entry from `PROJECT/.gitignore`. |
| `PROJECT/requirements.txt` | Duplicate dependency list; root `requirements.txt` is the documented installation source. `PROJECT/pyproject.toml` and `PROJECT/uv.lock` are retained for the existing Python/uv project metadata. |
| `PROJECT/public/` | Empty directory after removal of its placeholder file. |

## Removed regenerable artifacts

- `PROJECT/__pycache__/`
  - `evaluate.cpython-314.pyc`
  - `ocr.cpython-314.pyc`
  - `server.cpython-314.pyc`
  - `__init__.cpython-314.pyc`
- `tests/__pycache__/`
  - `test_api.cpython-314.pyc`
  - `test_integration.cpython-314.pyc`
  - `test_ocr.cpython-314.pyc`
- `PROJECT/.vercel/output/` — generated Vercel build output, not application source:
  - `builds.json`
  - `config.json`
  - `diagnostics/` (`cli_traces.json`, `deploy-manifest.json`, `project-manifest.json`)
  - `functions/`
    - `index.func/`
      - `.vc-config.json`
      - `vc__handler__python.py`
  - `static/` (`index.html`)
  - The `diagnostics/`, `functions/`, `functions/index.func/`, and `static/`
    directories were removed with the generated output.
- `.ocr-camera-probe.png`, `.ocr-camera-probe-preprocessed.png`, and
  `evaluation/samples/line_hello-preprocessed.png` — temporary debug images
  created for OCR false-positive/preprocessing checks; removed after testing.

## Preserved intentionally

- `.venv/` — verified active Python 3.14.2 environment containing the tested
  Flask, PyTorch, Transformers, and image dependencies. It was not deleted; the
  README documents recreation from root `requirements.txt`.
- `.git/`, `.vscode/`, local Vercel environment/project metadata, evaluation
  labels and samples, application source, and tests.
- Existing `reports/` files were preserved. The post-fix 100-sample IAM
  evaluation is saved separately under `reports/ocr-fix-evaluation/`.
- Root `vercel.json` and `api/index.py`; they were retained, but the README
  documents that Vercel serverless deployment has not been validated for the
  PyTorch/model runtime and requires the repository root as the project root.
