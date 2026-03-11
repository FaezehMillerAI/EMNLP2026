# EMLP XAI Report Generation (IU-Xray)

Lightweight, explainability-first pipeline for automated medical report generation without heatmaps.

## Quickstart (Colab)
1. Mount Drive and set output dir:
   ```bash
   export OUTPUT_DIR=/content/drive/MyDrive/EMLP_XAI
   ```
2. Install deps:
   ```bash
   pip install -r requirements.txt
   ```
3. Download IU-Xray from Kaggle (requires `~/.kaggle/kaggle.json`):
   ```bash
   python scripts/download_kaggle.py --out data/raw
   python scripts/prepare_iuxray.py --raw data/raw/iu-xray --out data/processed
   ```
4. Smoke test:
   ```bash
   python scripts/smoke_test.py --config src/configs/default.yaml --output_dir "$OUTPUT_DIR"
   ```

## Notes
- All outputs (checkpoints, metrics, qualitative figures) are saved under `output_dir`.
- Explainability outputs are text-first: concept evidence tables, counterfactual influence scores, and claim–concept alignments. No heatmaps are produced.

