# IAM Handwriting Dataset Cloud Streaming Evaluation Report

## Configuration & Environment
- **Dataset**: `Teklia/IAM-line` (accessed via HTTP streaming parquet, split: `test`)
- **Model**: `microsoft/trocr-small-handwritten`
- **Device Used**: `cpu`
- **Total Evaluated Samples**: 100
- **Streaming Mode**: Zero-disk dataset streaming (read on-the-fly via pyarrow & fsspec)

## Performance Metrics
| Metric | Value |
| :--- | :--- |
| **Total Samples** | 100 |
| **Successful Predictions** | 100 |
| **Failed Predictions** | 0 |
| **Character Error Rate (CER)** | 6.51% (0.0651) |
| **Word Error Rate (WER)** | 16.74% (0.1674) |
| **Exact-Match Accuracy** | 35.00% (0.3500) |
| **Average Inference Time** | 555.2 ms/sample |
| **Total Wall Clock Time** | 59.7 s |

## Sample Qualitative Results (First 5 Predictions)
| Index | Ground Truth | Prediction | CER | WER | Exact | Time (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | assuredness " Bella Bella Marie " ( Parlophone ) , a lively song that changes tempo mid-way . | assuredness . " Bella Bella Marie " ( Rarlophone ) , a lively song that changes tempo mid-way . | 0.0323 | 0.1111 | False | 1360.92 |
| 1 | I don't think he will storm the charts with this one , but it 's a good start . | I don't think he will storm the charts with this one , but it's a good start . | 0.0127 | 0.1053 | False | 1097.97 |
| 2 | CHRIS CHARLES , 39 , who lives in Stockton-on-Tees , is an accountant . | CHRISTS CHARLES , 39 , who lives in Stockton - on - Tees , is an accountant . | 0.0845 | 0.4286 | False | 1104.19 |
| 3 | Become a success with a disc and hey presto ! You 're a star ... . Rolly sings with | become a success with a disc and key pesto ! You're a star . . Rolly sings with | 0.0723 | 0.3158 | False | 1146.88 |
| 4 | Tolch , as he is known in Tin Pan Alley , likes songs with a month in the title . He wrote | Idlch , as he is known in Tin Pan Alley , lives songs with a month in the title . He wrote | 0.0333 | 0.0909 | False | 1230.64 |

## Errors Encountered
None. All samples evaluated successfully.
