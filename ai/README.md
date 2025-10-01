# AI-Based Cancer Detection from Chest X-rays

Main entry point: `ai/main.py`

## Quick Start

- **Create env (Conda)**
  - `conda env create -f environment.yml`
  - `conda activate cancer-detection`
- **Or install with pip**
  - `python -m venv .venv && .\.venv\Scripts\activate` (Windows)
  - `pip install -r requirements.txt`

## Dataset Layout

Update `configs/config.yaml` if your paths differ.

```
ai/
  data/
    raw/
      images/                         # Chest X-ray images (e.g., NIH ChestX-ray14)
      Data_Entry_2017_v2020.csv       # Labels CSV
    processed/
    train/
    val/
    test/
```

Minimum required for training/eval:
- `data/raw/images/` with image files
- `data/raw/Data_Entry_2017_v2020.csv` with columns `Image Index`, `Finding Labels`

## Commands

- **Train**
```
python ai/main.py train \
  --config ai/configs/config.yaml \
  --device auto \
  --experiment-name resnet50_baseline
```
This creates a timestamped folder under `ai/experiments/` with `checkpoints/`, `logs/`, `results/`, and a copy of the config.

- **Evaluate**
```
python ai/main.py evaluate \
  --config ai/configs/config.yaml \
  --checkpoint ai/experiments/<your_exp>/checkpoints/best_model.pth \
  --device auto
```
Saves metrics and plots to `ai/experiments/evaluation_results/` unless `--output` is provided.

- **Inference (single image)**
```
python ai/main.py inference \
  --config ai/configs/config.yaml \
  --checkpoint ai/experiments/<your_exp>/checkpoints/best_model.pth \
  --image path/to/image.png \
  --visualize \
  --device auto
```
Outputs prediction, confidence, optional Grad-CAM overlay and a JSON report.

- **Inference (batch)**
```
python ai/main.py inference \
  --config ai/configs/config.yaml \
  --checkpoint ai/experiments/<your_exp>/checkpoints/best_model.pth \
  --batch-images path/to/dir \
  --device auto
```
Creates `batch_predictions.json` under `ai/experiments/inference_results/`.

- **Demo**
```
python ai/main.py demo \
  --config ai/configs/config.yaml \
  --checkpoint ai/experiments/<your_exp>/checkpoints/best_model.pth
```
Looks for images under `ai/data/sample_images/`.

## Testing

Run unit tests (fast, uses temporary data):
```
python -m unittest discover -s ai/tests -p "test_*.py" -v
```

## Notes

- Models: `custom_cnn`, `resnet18/34/50/101`, `densenet121/161/169/201`, `efficientnet_b0/b1/b3`
- Binary classification with BCEWithLogitsLoss; positive class is cancer (`Mass`, `Nodule`).
- Class imbalance handled via `pos_weight` (computed from dataset at runtime).
