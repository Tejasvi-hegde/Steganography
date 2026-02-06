"""
Flask Backend API for Deep Learning Steganography
"""
import os
import sys
import io
import time
import base64
import logging

# Setup path before imports
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(ROOT_DIR, 'backend_debug.log')),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
import numpy as np

# For SSIM calculation
try:
    from skimage.metrics import structural_similarity as ssim
    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False
    logger.warning("scikit-image not available, SSIM will be approximated")

# For histogram generation
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available, histograms will be disabled")

# For PDF generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch
    from reportlab.lib.colors import HexColor
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logger.warning("reportlab not available, PDF reports will be disabled")

app = Flask(__name__)
CORS(app)

# Globals
_encoder = None
_decoder = None
_device = None
_transform = None
_torch = None
_loaded = False

IMG_SIZE = 256  # Match training size for better quality


def get_models():
    """Get or load models"""
    global _encoder, _decoder, _device, _transform, _torch, _loaded
    
    if _loaded:
        return _encoder, _decoder, _device, _transform, _torch
    
    import torch
    from torchvision import transforms
    from models.encoder import StegoEncoder
    from models.decoder import StegoDecoder
    
    _torch = torch
    _device = torch.device('cpu')  # Use CPU for stability
    logger.info(f"Using device: {_device}")
    
    _encoder = StegoEncoder().to(_device)
    _decoder = StegoDecoder().to(_device)
    
    # Load weights
    checkpoint_dir = os.path.join(ROOT_DIR, 'outputs', 'checkpoints')
    encoder_path = os.path.join(checkpoint_dir, 'encoder_final.pth')
    decoder_path = os.path.join(checkpoint_dir, 'decoder_final.pth')
    
    if os.path.exists(encoder_path) and os.path.exists(decoder_path):
        _encoder.load_state_dict(torch.load(encoder_path, map_location=_device))
        _decoder.load_state_dict(torch.load(decoder_path, map_location=_device))
        logger.info("Weights loaded!")
    else:
        logger.warning("No weights found, using random weights")
    
    _encoder.eval()
    _decoder.eval()
    
    _transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    _loaded = True
    return _encoder, _decoder, _device, _transform, _torch


def tensor_to_base64(tensor, torch_module):
    """Convert tensor to base64 PNG - ensures RGB color output"""
    # Denormalize from [-1, 1] to [0, 1]
    t = tensor * 0.5 + 0.5
    t = torch_module.clamp(t, 0, 1)
    
    # Convert to numpy: [C, H, W] -> [H, W, C]
    arr = t.cpu().detach().numpy()
    arr = np.transpose(arr, (1, 2, 0))  # [H, W, 3]
    arr = (arr * 255).astype(np.uint8)
    
    # Ensure it's contiguous and RGB
    arr = np.ascontiguousarray(arr)
    
    # Create RGB image
    img = Image.fromarray(arr, mode='RGB')
    
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return 'data:image/png;base64,' + base64.b64encode(buf.read()).decode()


def calculate_ssim(img1_tensor, img2_tensor, torch_module):
    """Calculate SSIM between two tensors"""
    # Denormalize
    img1 = img1_tensor * 0.5 + 0.5
    img2 = img2_tensor * 0.5 + 0.5
    img1 = torch_module.clamp(img1, 0, 1)
    img2 = torch_module.clamp(img2, 0, 1)
    
    # Convert to numpy
    arr1 = img1.cpu().detach().numpy().squeeze()
    arr2 = img2.cpu().detach().numpy().squeeze()
    
    if arr1.ndim == 3:
        arr1 = np.transpose(arr1, (1, 2, 0))
        arr2 = np.transpose(arr2, (1, 2, 0))
    
    if SSIM_AVAILABLE:
        # Use scikit-image SSIM
        return ssim(arr1, arr2, channel_axis=2 if arr1.ndim == 3 else None, data_range=1.0)
    else:
        # Simple approximation based on MSE
        mse = np.mean((arr1 - arr2) ** 2)
        return max(0, 1 - mse * 10)


def generate_histogram_comparison(cover_tensor, stego_tensor, torch_module):
    """Generate histogram comparison image as base64"""
    if not MATPLOTLIB_AVAILABLE:
        return None
    
    # Denormalize
    cover = cover_tensor * 0.5 + 0.5
    stego = stego_tensor * 0.5 + 0.5
    cover = torch_module.clamp(cover, 0, 1)
    stego = torch_module.clamp(stego, 0, 1)
    
    # Convert to numpy
    cover_arr = cover.cpu().detach().numpy().squeeze()
    stego_arr = stego.cpu().detach().numpy().squeeze()
    
    if cover_arr.ndim == 3:
        cover_arr = np.transpose(cover_arr, (1, 2, 0))
        stego_arr = np.transpose(stego_arr, (1, 2, 0))
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    colors = ['red', 'green', 'blue']
    channel_names = ['Red', 'Green', 'Blue']
    
    for i, (color, name) in enumerate(zip(colors, channel_names)):
        axes[i].hist(cover_arr[:,:,i].flatten(), bins=256, range=(0,1), 
                     alpha=0.5, label='Cover', color=color, histtype='step', linewidth=1.5)
        axes[i].hist(stego_arr[:,:,i].flatten(), bins=256, range=(0,1), 
                     alpha=0.5, label='Stego', color='black', histtype='step', linewidth=1.5, linestyle='--')
        axes[i].set_title(f'{name} Channel')
        axes[i].set_xlabel('Pixel Intensity')
        axes[i].set_ylabel('Frequency')
        axes[i].legend()
        axes[i].set_xlim(0, 1)
    
    plt.tight_layout()
    
    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    plt.close(fig)
    
    return 'data:image/png;base64,' + base64.b64encode(buf.read()).decode()


@app.route('/')
def index():
    return jsonify({'message': 'Steganography API', 'status': 'running'})


@app.route('/api/health')
def health():
    try:
        encoder, decoder, device, _, _ = get_models()
        return jsonify({
            'status': 'healthy',
            'model_loaded': True,
            'encoder_params': sum(p.numel() for p in encoder.parameters()),
            'decoder_params': sum(p.numel() for p in decoder.parameters()),
            'device': str(device)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/hide', methods=['POST'])
def hide():
    try:
        logger.info("=== /api/hide called ===")
        logger.info(f"Request files: {list(request.files.keys())}")
        
        encoder, decoder, device, transform, torch = get_models()
        logger.info("Models loaded successfully")
        
        if 'cover' not in request.files or 'secret' not in request.files:
            logger.error("Missing files in request")
            return jsonify({'success': False, 'error': 'Need cover and secret images'}), 400
        
        logger.info("Opening images...")
        cover = Image.open(request.files['cover']).convert('RGB')
        secret = Image.open(request.files['secret']).convert('RGB')
        logger.info(f"Cover size: {cover.size}, Secret size: {secret.size}")
        
        logger.info("Transforming images...")
        cover_t = transform(cover).unsqueeze(0).to(device)
        secret_t = transform(secret).unsqueeze(0).to(device)
        logger.info(f"Tensor shapes: cover={cover_t.shape}, secret={secret_t.shape}")
        
        start = time.time()
        
        logger.info("Running encoder and decoder...")
        with torch.no_grad():
            logger.info(f"Cover shape: {cover_t.shape}, Secret shape: {secret_t.shape}")
            stego_t = encoder(cover_t, secret_t)
            logger.info(f"Stego shape: {stego_t.shape}")
            recovered_t = decoder(stego_t)
            logger.info(f"Recovered shape: {recovered_t.shape}")
        
        proc_time = time.time() - start
        logger.info(f"Processing time: {proc_time:.2f}s")
        
        # Metrics
        cover_d = cover_t * 0.5 + 0.5
        stego_d = stego_t * 0.5 + 0.5
        secret_d = secret_t * 0.5 + 0.5
        recovered_d = recovered_t * 0.5 + 0.5
        
        mse_stego = torch.mean((cover_d - stego_d) ** 2).item()
        mse_rec = torch.mean((secret_d - recovered_d) ** 2).item()
        
        psnr_stego = 10 * np.log10(1.0 / max(mse_stego, 1e-10))
        psnr_rec = 10 * np.log10(1.0 / max(mse_rec, 1e-10))
        
        # Calculate SSIM
        ssim_stego = calculate_ssim(cover_t, stego_t, torch)
        ssim_recovery = calculate_ssim(secret_t, recovered_t, torch)
        
        # Generate histogram comparison
        histogram_b64 = generate_histogram_comparison(cover_t, stego_t, torch)
        
        logger.info("Converting tensors to base64...")
        stego_b64 = tensor_to_base64(stego_t[0], torch)
        recovered_b64 = tensor_to_base64(recovered_t[0], torch)
        logger.info(f"Base64 lengths: stego={len(stego_b64)}, recovered={len(recovered_b64)}")
        
        result = {
            'success': True,
            'data': {
                'stegoImage': stego_b64,
                'recoveredSecret': recovered_b64,
                'metrics': {
                    'psnrStego': round(psnr_stego, 2),
                    'ssimStego': round(ssim_stego, 4),
                    'psnrRecovery': round(psnr_rec, 2),
                    'ssimRecovery': round(ssim_recovery, 4),
                    'mse': round(mse_rec, 6),
                    'processingTime': round(proc_time, 2)
                },
                'histogramComparison': histogram_b64
            }
        }
        logger.info("Success! Returning response")
        return jsonify(result)
    except Exception as e:
        import traceback
        logger.error("=== ERROR IN /api/hide ===")
        logger.error(traceback.format_exc())
        logger.error(f"Error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/extract', methods=['POST'])
def extract():
    try:
        _, decoder, device, transform, torch = get_models()
        
        if 'stego' not in request.files:
            return jsonify({'success': False, 'error': 'Need stego image'}), 400
        
        stego = Image.open(request.files['stego']).convert('RGB')
        stego_t = transform(stego).unsqueeze(0).to(device)
        
        start = time.time()
        
        with torch.no_grad():
            recovered_t = decoder(stego_t)
        
        proc_time = time.time() - start
        
        return jsonify({
            'success': True,
            'data': {
                'recoveredSecret': tensor_to_base64(recovered_t[0], torch),
                'processingTime': round(proc_time, 2)
            }
        })
    except Exception as e:
        import traceback
        logger.error("=== ERROR IN /api/extract ===")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/robustness-test', methods=['POST'])
def robustness_test():
    """Test robustness of stego image against JPEG compression"""
    try:
        _, decoder, device, transform, torch = get_models()
        
        if 'stego' not in request.files:
            return jsonify({'success': False, 'error': 'Need stego image'}), 400
        
        # Get the original stego image
        stego_img = Image.open(request.files['stego']).convert('RGB')
        
        # Test different JPEG quality levels
        quality_levels = [95, 85, 75, 50]
        results = []
        
        for quality in quality_levels:
            # Compress as JPEG
            buf = io.BytesIO()
            stego_img.save(buf, format='JPEG', quality=quality)
            buf.seek(0)
            compressed_img = Image.open(buf).convert('RGB')
            
            # Transform and extract
            compressed_t = transform(compressed_img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                recovered_t = decoder(compressed_t)
            
            # Calculate metrics (compare with original stego extraction)
            original_t = transform(stego_img).unsqueeze(0).to(device)
            with torch.no_grad():
                original_recovered = decoder(original_t)
            
            # SSIM between original recovery and compressed recovery
            recovery_ssim = calculate_ssim(original_recovered, recovered_t, torch)
            
            results.append({
                'quality': quality,
                'recoveredImage': tensor_to_base64(recovered_t[0], torch),
                'ssim': round(recovery_ssim, 4)
            })
        
        return jsonify({
            'success': True,
            'data': {
                'results': results
            }
        })
    except Exception as e:
        import traceback
        logger.error("=== ERROR IN /api/robustness-test ===")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/generate-report', methods=['POST'])
def generate_report():
    """Generate a PDF report of the steganography analysis"""
    try:
        if not REPORTLAB_AVAILABLE:
            return jsonify({'success': False, 'error': 'PDF generation not available'}), 500
        
        # Get metrics from request
        data = request.get_json()
        metrics = data.get('metrics', {})
        
        # Create PDF
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=A4)
        width, height = A4
        
        # Title
        c.setFont("Helvetica-Bold", 24)
        c.drawString(50, height - 50, "Steganography Analysis Report")
        
        # Subtitle
        c.setFont("Helvetica", 12)
        c.drawString(50, height - 75, f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Line separator
        c.setStrokeColor(HexColor('#00D9C0'))
        c.setLineWidth(2)
        c.line(50, height - 90, width - 50, height - 90)
        
        # Model Information
        y = height - 130
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, y, "Model Architecture")
        
        y -= 25
        c.setFont("Helvetica", 11)
        model_info = [
            "• Encoder-Decoder GAN Architecture",
            "• Squeeze-and-Excitation (SE) Attention Blocks",
            "• U-Net Style Skip Connections",
            "• Total Parameters: ~6.2M (Encoder: 3.1M, Decoder: 3.1M)"
        ]
        for info in model_info:
            c.drawString(70, y, info)
            y -= 18
        
        # Quality Metrics
        y -= 20
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, y, "Quality Metrics")
        
        y -= 30
        c.setFont("Helvetica", 11)
        
        # Stego Quality
        c.setFont("Helvetica-Bold", 12)
        c.drawString(70, y, "Stego Image Quality (Cover vs Stego):")
        y -= 20
        c.setFont("Helvetica", 11)
        c.drawString(90, y, f"PSNR: {metrics.get('psnrStego', 'N/A')} dB")
        y -= 18
        c.drawString(90, y, f"SSIM: {metrics.get('ssimStego', 'N/A')}")
        
        y -= 30
        c.setFont("Helvetica-Bold", 12)
        c.drawString(70, y, "Recovery Quality (Original Secret vs Recovered):")
        y -= 20
        c.setFont("Helvetica", 11)
        c.drawString(90, y, f"PSNR: {metrics.get('psnrRecovery', 'N/A')} dB")
        y -= 18
        c.drawString(90, y, f"SSIM: {metrics.get('ssimRecovery', 'N/A')}")
        y -= 18
        c.drawString(90, y, f"MSE: {metrics.get('mse', 'N/A')}")
        
        # Processing Info
        y -= 40
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, y, "Processing Information")
        
        y -= 25
        c.setFont("Helvetica", 11)
        c.drawString(70, y, f"Processing Time: {metrics.get('processingTime', 'N/A')} seconds")
        y -= 18
        c.drawString(70, y, "Image Size: 256 x 256 pixels")
        y -= 18
        c.drawString(70, y, "Device: CPU")
        
        # Quality Assessment
        y -= 40
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, y, "Quality Assessment")
        
        y -= 25
        psnr = metrics.get('psnrStego', 0)
        ssim_val = metrics.get('ssimStego', 0)
        
        if psnr >= 30 and ssim_val >= 0.95:
            quality = "EXCELLENT"
            assessment = "The stego image is virtually indistinguishable from the original."
        elif psnr >= 25 and ssim_val >= 0.85:
            quality = "GOOD"
            assessment = "Good quality steganography with minimal visible artifacts."
        else:
            quality = "FAIR"
            assessment = "Acceptable quality. Some artifacts may be visible."
        
        c.setFont("Helvetica-Bold", 14)
        c.setFillColor(HexColor('#00D9C0'))
        c.drawString(70, y, f"Overall Quality: {quality}")
        c.setFillColor(HexColor('#000000'))
        
        y -= 20
        c.setFont("Helvetica", 11)
        c.drawString(70, y, assessment)
        
        # Footer
        c.setFont("Helvetica-Oblique", 9)
        c.drawString(50, 30, "Generated by Deep Learning Steganography System")
        
        c.save()
        buf.seek(0)
        
        return send_file(
            buf,
            mimetype='application/pdf',
            as_attachment=True,
            download_name='steganography_report.pdf'
        )
    except Exception as e:
        import traceback
        logger.error("=== ERROR IN /api/generate-report ===")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    logger.info("=" * 50)
    logger.info("Steganography API Server")
    logger.info("=" * 50)
    
    # Pre-load models
    logger.info("Loading models...")
    try:
        get_models()
        logger.info("Models ready!")
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
    
    logger.info("\nServer: http://localhost:5000")
    logger.info("=" * 50)
    
    app.run(host='127.0.0.1', port=5000, debug=False)
