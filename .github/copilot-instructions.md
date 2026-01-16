# Copilot Instructions for Steganography Project

This project implements a Deep Learning based Image Steganography system using PyTorch (with DirectML support). It hides a "secret" image inside a "cover" image using an Encoder-Decoder GAN architecture.

## Project Architecture & Overview
- **Core Logic**: GAN-based Steganography (Encoder hides secret, Decoder recovers it).
- **Structure**:
  - `models/`: PyTorch modules (`StegoEncoder`, `StegoDecoder`, `Discriminator`).
  - `scripts/`: Execution entry points. **Scripts are designed to be run from inside the `scripts/` directory** due to relative data/output paths (e.g., `../outputs`).
  - `data/`: Dataset storage. Standard layout: `cover/` and `secret/` subdirectories.
  - `outputs/`: Artifacts. Checkpoints go to `checkpoints/`, logs to `tensorboard/`.
- **Key Files**:
  - `scripts/train.py`: Main training loop.
  - `scripts/hide.py`: Inference script to hide an image.
  - `scripts/extract.py`: Inference script to recover an image.
  - `utils/dataset.py`: `StegoDataset` data loader class.

## Critical Developer Workflows

### 1. Environment & Setup
- Use `torch-directml` for GPU acceleration on Windows:
  ```bash
  pip install torch-directml
  ```
- Install dependencies from `requirements.txt`.

### 2. Training
- **Always change directory to `scripts/` first**:
  ```bash
  cd scripts
  python train.py
  ```
- This ensures relative paths like `../outputs` and `../data` resolve correctly.
- Default device selection prefers `cuda` (which DirectML maps to), falling back to `cpu`.

### 3. Inference
- To hide a secret image:
  ```bash
  cd scripts
  python hide.py --cover ../data/test/cover.jpg --secret ../data/test/secret.jpg --output ../outputs/stego.png --encoder ../outputs/checkpoints/encoder_final.pth
  ```
- To recover a secret image:
  ```bash
  cd scripts
  python extract.py --stego ../outputs/stego.png --output ../outputs/recovered.png --decoder ../outputs/checkpoints/decoder_final.pth
  ```

### 4. Data Preparation
- Use `scripts/download_coco.py` to fetch COCO dataset.
- Use `scripts/prepare_coco_for_training.py` to organize images into `cover` and `secret` folders.
- Dataset classes (`StegoDataset`) expect `cover` and `secret` directories containing images.

## Coding Conventions
- **Imports**: Scripts invoke `sys.path.append('..')` to import from `models` and `utils` modules located in the project root. Preserve this pattern in new scripts.
- **Image Normalization**: Images are normalized to `[-1, 1]` (mean=[0.5]*3, std=[0.5]*3). Adjust accordingly when visualizing or saving (e.g., `x * 0.5 + 0.5`).
- **Paths**: Use relative paths starting with `../` when referencing project root directories from within `scripts/`.

## Common Pitfalls
- **WorkingDirectory**: Running scripts from the project root (e.g., `python scripts/train.py`) will likely fail or misplace outputs because code uses relative `../` paths requiring `scripts/` as CWD.
- **Model Matching**: Ensure the saved model paths strictly correspond to the architecture defined in `models/`.
