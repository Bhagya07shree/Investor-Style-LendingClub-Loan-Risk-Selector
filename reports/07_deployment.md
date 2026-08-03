# Deployment — Pipeline Architecture, App, and Known Limitations

## 1. Pipeline Architecture

The trained model is served through a single, reproducible preprocessing pipeline shared between training and inference — this is deliberate, and central to avoiding train/serve mismatch.

| File | Role |
|---|---|
| `src/pipeline/preprocess.py` | Single source of truth for every transformation (date parsing, feature engineering, capping, imputation, encoding). Used identically by both `train.py` and `predict.py`. |
| `src/pipeline/train.py` | Reproduces the final model end-to-end from raw CSV with one command. |
| `src/pipeline/predict.py` | Inference entry point — loads the saved model and artifacts, scores new applications using the exact same preprocessing logic as training. |
| `src/pipeline/monitor_drift.py` | PSI-based drift monitoring (see `05_feature_drift_analysis.md`). |
| `src/artifacts/` | All fitted preprocessing objects — imputers, encoder, winsorization caps, category mappings — fit once on training data and only ever *applied*, never re-fit, at inference time. |

**Why this matters:** because `preprocess.py` is imported by both training and inference, there is no
