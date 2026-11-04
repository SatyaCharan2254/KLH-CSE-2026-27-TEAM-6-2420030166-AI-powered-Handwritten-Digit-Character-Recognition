"""Stream evaluation for IAM dataset using Teklia/IAM-line.

Evaluates microsoft/trocr-small-handwritten without downloading the complete dataset.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Generator, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import fsspec
from PIL import Image
import pyarrow.parquet as pq

from PROJECT.ocr import (
    MODEL_ID,
    character_error_rate,
    load_model,
    recognize_image,
    word_error_rate,
)

logger = logging.getLogger(__name__)

DATASET_REPO = "Teklia/IAM-line"
DEFAULT_SPLIT = "test"
SPLIT_URL_TEMPLATE = (
    "https://huggingface.co/datasets/Teklia/IAM-line/resolve/main/data/{split}.parquet"
)


@dataclass
class StreamSample:
    index: int
    split: str
    ground_truth: str
    image: Image.Image


def stream_iam_samples(
    split: str = DEFAULT_SPLIT,
    limit: Optional[int] = 100,
    hf_token: Optional[str] = None,
) -> Generator[StreamSample, None, None]:
    """Yield samples one-by-one directly from cloud parquet without full download."""
    url = SPLIT_URL_TEMPLATE.format(split=split)
    headers = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    fs, path = fsspec.core.url_to_fs(url, headers=headers)
    yielded = 0

    with fs.open(path, "rb") as f:
        parquet_file = pq.ParquetFile(f)
        num_row_groups = parquet_file.num_row_groups

        for rg_idx in range(num_row_groups):
            row_group = parquet_file.read_row_group(rg_idx)
            text_col = row_group["text"]
            image_col = row_group["image"]

            for row_idx in range(len(row_group)):
                ground_truth = text_col[row_idx].as_py()
                image_dict = image_col[row_idx].as_py()
                raw_bytes = image_dict.get("bytes") if isinstance(image_dict, dict) else None

                if raw_bytes is None:
                    continue

                image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                sample = StreamSample(
                    index=yielded,
                    split=split,
                    ground_truth=ground_truth,
                    image=image,
                )
                yield sample
                yielded += 1

                if limit is not None and yielded >= limit:
                    return


def run_cloud_evaluation(
    limit: int = 100,
    split: str = "test",
    output_dir: Path | str = "reports",
) -> dict[str, Any]:
    """Stream samples, run inference, score metrics, and save output files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    _, model, device = load_model()
    actual_device = str(device)
    model_device_attr = str(getattr(model, "device", device))

    logger.info("Running evaluation on %s (model device: %s)", actual_device, model_device_attr)

    predictions_records: list[dict[str, Any]] = []
    cer_list: list[float] = []
    wer_list: list[float] = []
    exact_matches = 0
    inference_times_ms: list[float] = []
    errors_list: list[dict[str, Any]] = []

    start_eval_time = time.perf_counter()

    for sample in stream_iam_samples(split=split, limit=limit):
        t0 = time.perf_counter()
        error_msg = None
        prediction = ""
        cer = 1.0
        wer = 1.0
        is_exact = False

        try:
            prediction = recognize_image(sample.image).extracted_text
            inference_ms = (time.perf_counter() - t0) * 1000
            inference_times_ms.append(inference_ms)

            cer = character_error_rate(sample.ground_truth, prediction)
            wer = word_error_rate(sample.ground_truth, prediction)
            is_exact = (sample.ground_truth.strip() == prediction.strip())

            cer_list.append(cer)
            wer_list.append(wer)
            if is_exact:
                exact_matches += 1

        except Exception as exc:
            inference_ms = (time.perf_counter() - t0) * 1000
            error_msg = f"{exc.__class__.__name__}: {exc}"
            errors_list.append({
                "sample_index": sample.index,
                "ground_truth": sample.ground_truth,
                "error": error_msg,
            })
            logger.warning("Error predicting sample %d: %s", sample.index, error_msg)

        record = {
            "sample_index": sample.index,
            "split": sample.split,
            "ground_truth": sample.ground_truth,
            "prediction": prediction,
            "cer": round(cer, 4) if error_msg is None else None,
            "wer": round(wer, 4) if error_msg is None else None,
            "exact_match": is_exact,
            "inference_time_ms": round(inference_ms, 2),
            "error": error_msg,
        }
        predictions_records.append(record)

    total_evaluated = len(predictions_records)
    successful_count = len(cer_list)
    total_duration_sec = time.perf_counter() - start_eval_time

    mean_cer = float(sum(cer_list) / len(cer_list)) if cer_list else 0.0
    mean_wer = float(sum(wer_list) / len(wer_list)) if wer_list else 0.0
    exact_match_acc = float(exact_matches / successful_count) if successful_count else 0.0
    avg_inference_time_ms = (
        float(sum(inference_times_ms) / len(inference_times_ms))
        if inference_times_ms
        else 0.0
    )

    summary = {
        "dataset": DATASET_REPO,
        "split": split,
        "model_id": MODEL_ID,
        "device": actual_device,
        "model_device_attribute": model_device_attr,
        "total_evaluated_samples": total_evaluated,
        "successful_evaluations": successful_count,
        "failed_evaluations": len(errors_list),
        "mean_cer": round(mean_cer, 4),
        "mean_wer": round(mean_wer, 4),
        "exact_match_accuracy": round(exact_match_acc, 4),
        "average_inference_time_ms": round(avg_inference_time_ms, 2),
        "total_wall_time_seconds": round(total_duration_sec, 2),
        "errors": errors_list,
    }

    # 1. Write reports/evaluation_summary.json
    summary_path = output_path / "evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # 2. Write reports/iam_predictions.csv
    csv_path = output_path / "iam_predictions.csv"
    if predictions_records:
        keys = [
            "sample_index",
            "split",
            "ground_truth",
            "prediction",
            "cer",
            "wer",
            "exact_match",
            "inference_time_ms",
            "error",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=keys)
            writer.writeheader()
            writer.writerows(predictions_records)

    # 3. Write reports/evaluation_report.md
    report_md_path = output_path / "evaluation_report.md"
    report_content = f"""# IAM Handwriting Dataset Cloud Streaming Evaluation Report

## Configuration & Environment
- **Dataset**: `{DATASET_REPO}` (accessed via HTTP streaming parquet, split: `{split}`)
- **Model**: `{MODEL_ID}`
- **Device Used**: `{actual_device}`
- **Total Evaluated Samples**: {total_evaluated}
- **Streaming Mode**: Zero-disk dataset streaming (read on-the-fly via pyarrow & fsspec)

## Performance Metrics
| Metric | Value |
| :--- | :--- |
| **Total Samples** | {total_evaluated} |
| **Successful Predictions** | {successful_count} |
| **Failed Predictions** | {len(errors_list)} |
| **Character Error Rate (CER)** | {mean_cer * 100:.2f}% ({mean_cer:.4f}) |
| **Word Error Rate (WER)** | {mean_wer * 100:.2f}% ({mean_wer:.4f}) |
| **Exact-Match Accuracy** | {exact_match_acc * 100:.2f}% ({exact_match_acc:.4f}) |
| **Average Inference Time** | {avg_inference_time_ms:.1f} ms/sample |
| **Total Wall Clock Time** | {total_duration_sec:.1f} s |

## Sample Qualitative Results (First 5 Predictions)
| Index | Ground Truth | Prediction | CER | WER | Exact | Time (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for rec in predictions_records[:5]:
        gt_esc = rec["ground_truth"].replace("|", "\\|")
        pred_esc = str(rec["prediction"]).replace("|", "\\|")
        report_content += f"| {rec['sample_index']} | {gt_esc} | {pred_esc} | {rec['cer']} | {rec['wer']} | {rec['exact_match']} | {rec['inference_time_ms']} |\n"

    if errors_list:
        report_content += "\n## Errors Encountered\n"
        for err in errors_list:
            report_content += f"- Sample {err['sample_index']}: {err['error']}\n"
    else:
        report_content += "\n## Errors Encountered\nNone. All samples evaluated successfully.\n"

    report_md_path.write_text(report_content, encoding="utf-8")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run cloud streaming evaluation on IAM dataset.")
    parser.add_argument("--limit", type=int, default=100, help="Number of samples to evaluate.")
    parser.add_argument("--split", type=str, default="test", help="Dataset split (test, validation, train).")
    parser.add_argument("--output-dir", type=str, default="reports", help="Output directory for reports.")
    args = parser.parse_args()

    res = run_cloud_evaluation(limit=args.limit, split=args.split, output_dir=args.output_dir)
    print("\nEvaluation Complete!")
    print(f"Evaluated: {res['total_evaluated_samples']} samples")
    print(f"Mean CER: {res['mean_cer'] * 100:.2f}%")
    print(f"Mean WER: {res['mean_wer'] * 100:.2f}%")
    print(f"Exact Match Accuracy: {res['exact_match_accuracy'] * 100:.2f}%")
    print(f"Avg Inference Time: {res['average_inference_time_ms']} ms/sample")
