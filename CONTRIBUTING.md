# Contributing to Steganography

We welcome contributions! Here's how to get started:

## Getting Started

1. **Fork the repository**
2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Making Changes

1. **Create a feature branch**:

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** and test them

3. **Keep commits clean**:
   - Write descriptive commit messages
   - Keep commits focused on single issues

## Code Style

- Follow PEP 8 conventions
- Use meaningful variable names
- Add docstrings to functions and classes

## Testing

Always test your changes before submitting:

```bash
cd scripts
python train.py  # Test training
python hide.py --cover test.jpg --secret test.jpg --output test_stego.png --encoder ../outputs/checkpoints/encoder_final.pth
python extract.py --stego test_stego.png --output test_recovered.png --decoder ../outputs/checkpoints/decoder_final.pth
```

## Submitting Changes

1. **Push to your fork**:

   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create a Pull Request** with a clear description of:
   - What changes you made
   - Why you made them
   - Any related issues

## Reporting Issues

When reporting issues, please include:

- Python version
- PyTorch version
- OS (Windows/Linux/Mac)
- Minimal reproducible example
- Error messages/logs

## Questions?

Feel free to open an issue for questions or discussions.
