# Steganography - Deep Learning Image Hiding

A PyTorch-based GAN implementation for hiding secret images inside cover images using neural networks.

## Overview

This project implements a **Generative Adversarial Network (GAN)** for image steganography. It consists of three main components:

- **Encoder**: Hides a secret image within a cover image, producing a stego image that appears indistinguishable from the original cover
- **Decoder**: Recovers the secret image from the stego image
- **Discriminator**: Ensures stego images are imperceptible by adversarially training the encoder

The system uses residual blocks and convolutional networks to learn robust image hiding while maintaining visual quality of the cover image.

## Requirements

- Python 3.8+
- PyTorch 2.0+ (CPU or GPU)
- See `requirements.txt` for dependencies

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd steganography

# Install dependencies
pip install -r requirements.txt
```

## Quick Start (CPU Training - Recommended for Laptops)

### 1. Download Dataset (Lightweight - ~50MB instead of 18GB COCO)

```bash
cd scripts

# EASIEST: Download random images from picsum.photos
python download_dataset.py --dataset picsum --num_images 1000 --max_pairs 400

# This downloads ~1000 images and creates 400 cover/secret pairs
# Uses DIFFERENT images for cover and secret (important for training!)
```

### 2. Train the Model (CPU Optimized)

```bash
python simple_train.py
```

Training settings (optimized for CPU):
- **Batch size**: 2 (memory efficient)
- **Epochs**: 20
- **Image size**: 128×128
- **Time**: ~30-60 minutes on modern CPU

### 3. Test the Model

```bash
python quick_test.py
```

This shows visual results with PSNR metrics.

### 4. Hide a Secret Image

```bash
python hide.py \
  --cover ../data/train/cover/image_00001.png \
  --secret ../data/train/secret/image_00002.png \
  --output ../outputs/stego.png \
  --encoder ../outputs/checkpoints/encoder_final.pth \
  --device cpu
```

### 5. Extract the Secret Image

```bash
python extract.py \
  --stego ../outputs/stego.png \
  --output ../outputs/recovered.png \
  --decoder ../outputs/checkpoints/decoder_final.pth \
  --device cpu
```

## Dataset Options

| Dataset | Size | Quality | Command |
|---------|------|---------|---------|
| **Picsum** (Recommended) | ~50MB | Good | `--dataset picsum --num_images 1000` |
| STL-10 | ~2.5GB | High | `--dataset stl10` |
| Flowers102 | ~350MB | Good | `--dataset flowers` |
| Custom | Varies | - | `--dataset custom --custom_dir /path/to/images` |

## Project Structure

```
steganography/
├── models/                    # Neural network architectures
│   ├── encoder.py            # StegoEncoder with residual blocks
│   ├── decoder.py            # StegoDecoder
│   └── discriminator.py      # Adversarial discriminator
├── scripts/                   # Entry points (run from this directory!)
│   ├── train.py              # Main training loop
│   ├── hide.py               # Inference: hide secret image
│   ├── extract.py            # Inference: extract secret image
│   ├── download_coco.py      # COCO dataset downloader
│   └── prepare_coco_for_training.py
├── utils/                     # Utility modules
│   ├── dataset.py            # StegoDataset class
│   ├── losses.py             # Custom loss functions
│   └── matrix.py             # Utility functions
├── data/                      # Dataset storage
│   ├── coco/                 # COCO images (after download)
│   └── train/                # Training pairs
├── outputs/                   # Generated artifacts
│   ├── checkpoints/          # Model weights
│   └── tensorboard/          # Training logs
└── README.md
```

## Important Notes

⚠️ **Working Directory**: Always run scripts from the `scripts/` directory:

```bash
cd scripts
python train.py
```

Running from the project root will fail because scripts use relative paths like `../outputs`.

## Image Normalization

Images are normalized to `[-1, 1]` range using:

```python
mean=[0.5, 0.5, 0.5]
std=[0.5, 0.5, 0.5]
```

When saving images, denormalize with:

```python
image = image * 0.5 + 0.5
```

## Architecture Details

### StegoEncoder

- Concatenates cover and secret images (6 channels input)
- Downsampling convolution
- Residual blocks for feature extraction
- Upsampling to original size
- Output: stego image (3 channels)

### StegoDecoder

- Takes stego image as input
- Feature extraction via convolutions
- Residual blocks
- Outputs recovered secret image

### Training Loop

- Generator loss: Reconstruction + adversarial + quality loss
- Discriminator loss: Distinguishes real from stego images
- Adam optimizer with β₁=0.5, β₂=0.999

## Monitoring Training

TensorBoard logs are saved to `../outputs/tensorboard/`:

```bash
tensorboard --logdir=../outputs/tensorboard/
```

## Custom Datasets

To use your own images, organize them as:

```
data/
├── cover/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── secret/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

Then update paths in training scripts.

## License

This project is provided as-is for educational purposes.

## Citation

If you use this project in research, please cite:

```bibtex
@software{steganography2026,
  title={Deep Learning Image Steganography with GANs},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/steganography}
}
```
