import unittest
from unittest.mock import MagicMock, patch
import io
import tempfile
from PIL import Image

from PROJECT.stream_evaluate import (
    StreamSample,
    run_cloud_evaluation,
    stream_iam_samples,
)


class StreamEvaluationTests(unittest.TestCase):
    def test_stream_sample_dataclass(self):
        img = Image.new("RGB", (100, 30), color=(255, 255, 255))
        sample = StreamSample(
            index=0,
            split="test",
            ground_truth="sample handwritten line",
            image=img,
        )
        self.assertEqual(sample.index, 0)
        self.assertEqual(sample.split, "test")
        self.assertEqual(sample.ground_truth, "sample handwritten line")
        self.assertEqual(sample.image.size, (100, 30))

    @patch("PROJECT.stream_evaluate.recognize_image")
    @patch("PROJECT.stream_evaluate.stream_iam_samples")
    @patch("PROJECT.stream_evaluate.load_model", return_value=(object(), object(), "cpu"))
    def test_cloud_evaluation_uses_final_ocr_text(
        self, _mock_load_model, mock_stream, mock_recognize
    ):
        mock_stream.return_value = iter(
            [
                StreamSample(
                    index=0,
                    split="test",
                    ground_truth="hello",
                    image=Image.new("RGB", (100, 30), (255, 255, 255)),
                )
            ]
        )
        mock_recognize.return_value = MagicMock(extracted_text="")

        with tempfile.TemporaryDirectory() as output_dir:
            summary = run_cloud_evaluation(limit=1, output_dir=output_dir)

        self.assertEqual(summary["mean_cer"], 1.0)
        self.assertEqual(summary["mean_wer"], 1.0)
        self.assertTrue(_mock_load_model.called)
        self.assertEqual(mock_recognize.call_count, 1)

    @patch("fsspec.core.url_to_fs")
    @patch("pyarrow.parquet.ParquetFile")
    def test_stream_mock_samples(self, mock_parquet_file_cls, mock_url_to_fs):
        mock_fs = MagicMock()
        mock_url_to_fs.return_value = (mock_fs, "fake/path.parquet")

        # Create dummy image bytes
        buf = io.BytesIO()
        Image.new("RGB", (50, 20)).save(buf, format="PNG")
        raw_bytes = buf.getvalue()

        # Mock row group
        mock_rg = MagicMock()
        mock_rg.__len__.return_value = 2
        mock_rg.__getitem__.side_effect = lambda col: (
            [MagicMock(as_py=lambda: "text 1"), MagicMock(as_py=lambda: "text 2")]
            if col == "text"
            else [MagicMock(as_py=lambda: {"bytes": raw_bytes}), MagicMock(as_py=lambda: {"bytes": raw_bytes})]
        )

        mock_pf_inst = MagicMock()
        mock_pf_inst.num_row_groups = 1
        mock_pf_inst.read_row_group.return_value = mock_rg
        mock_parquet_file_cls.return_value = mock_pf_inst

        samples = list(stream_iam_samples(split="test", limit=2))
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0].ground_truth, "text 1")
        self.assertEqual(samples[1].ground_truth, "text 2")
        self.assertEqual(samples[0].image.size, (50, 20))


if __name__ == "__main__":
    unittest.main()
