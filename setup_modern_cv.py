#!/usr/bin/env python3
"""
Setup script for Modern Autonomous Driving System
Using YOLOv8 + Semantic Segmentation

This script:
1. Installs required dependencies
2. Downloads pre-trained models
3. Sets up the environment for optimal performance
4. Validates the installation
"""

import os
import sys
import subprocess
import urllib.request
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        return False
    logger.info(f"Python version: {sys.version}")
    return True

def install_dependencies():
    """Install required Python packages."""
    logger.info("Installing dependencies...")
    
    try:
        # Upgrade pip first
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install PyTorch with CUDA support if available
        logger.info("Installing PyTorch...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu118"
        ])
        
        # Install other requirements
        if os.path.exists("requirements.txt"):
            logger.info("Installing from requirements.txt...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        else:
            logger.warning("requirements.txt not found, installing core packages...")
            core_packages = [
                "ultralytics>=8.0.0",
                "segmentation-models-pytorch>=0.3.0",
                "opencv-python>=4.5.0",
                "numpy>=1.21.0",
                "scikit-learn>=1.0.0",
                "albumentations>=1.3.0",
                "matplotlib>=3.5.0",
                "Pillow>=8.3.0",
                "tqdm>=4.62.0",
                "pyyaml>=6.0"
            ]
            
            for package in core_packages:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        
        logger.info("✅ Dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Error installing dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    directories = [
        "models",
        "data",
        "logs",
        "outputs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"📁 Created directory: {directory}")

def download_models():
    """Download pre-trained models."""
    logger.info("Downloading pre-trained models...")
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # YOLOv8 models will be downloaded automatically by ultralytics
    # when first used, but we can pre-download them
    
    yolo_models = [
        "yolov8n.pt",  # Nano - fastest
        "yolov8s.pt",  # Small - good balance
        "yolov8m.pt"   # Medium - better accuracy
    ]
    
    try:
        from ultralytics import YOLO
        
        for model_name in yolo_models:
            logger.info(f"Downloading {model_name}...")
            model = YOLO(model_name)  # This will download if not present
            logger.info(f"✅ {model_name} ready")
        
        logger.info("✅ YOLO models downloaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error downloading YOLO models: {e}")
        return False

def check_gpu_support():
    """Check for GPU support."""
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            logger.info(f"🚀 GPU Support: {gpu_count} GPU(s) available")
            logger.info(f"   Primary GPU: {gpu_name}")
            logger.info(f"   CUDA Version: {torch.version.cuda}")
            return True
        else:
            logger.warning("⚠️  No GPU support detected. System will use CPU only.")
            logger.info("   For better performance, consider installing CUDA and PyTorch with GPU support")
            return False
            
    except ImportError:
        logger.error("❌ PyTorch not installed properly")
        return False

def validate_installation():
    """Validate that all components are working."""
    logger.info("Validating installation...")
    
    try:
        # Test imports
        import cv2
        import numpy as np
        import torch
        from ultralytics import YOLO
        import segmentation_models_pytorch as smp
        
        logger.info("✅ All core packages imported successfully")
        
        # Test YOLO model loading
        model = YOLO('yolov8n.pt')
        logger.info("✅ YOLOv8 model loaded successfully")
        
        # Test segmentation model creation
        seg_model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=4
        )
        logger.info("✅ Segmentation model created successfully")
        
        # Test OpenCV
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.imwrite("test_image.jpg", test_image)
        os.remove("test_image.jpg")
        logger.info("✅ OpenCV working correctly")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        return False

def print_system_info():
    """Print system information."""
    logger.info("\n" + "="*60)
    logger.info("🚗 MODERN AUTONOMOUS DRIVING SYSTEM")
    logger.info("   Computer Vision Setup Complete")
    logger.info("="*60)
    
    try:
        import torch
        import cv2
        from ultralytics import YOLO
        
        logger.info(f"Python: {sys.version}")
        logger.info(f"PyTorch: {torch.__version__}")
        logger.info(f"OpenCV: {cv2.__version__}")
        logger.info(f"CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        
        logger.info("\n📋 System Components:")
        logger.info("   ✅ YOLOv8 for object detection")
        logger.info("   ✅ Semantic segmentation for lanes")
        logger.info("   ✅ Modern vehicle controller")
        logger.info("   ✅ Enhanced CARLA connector")
        
        logger.info("\n🎯 Usage:")
        logger.info("   1. Start CARLA simulator")
        logger.info("   2. Run: python src/main.py")
        logger.info("   3. Use controls: q=quit, r=reset, s=stats, p=performance, h=help")
        
    except Exception as e:
        logger.error(f"Error getting system info: {e}")

def main():
    """Main setup function."""
    print("🚗 Modern Autonomous Driving System - Setup")
    print("="*50)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        logger.error("Failed to install dependencies")
        return 1
    
    # Download models
    if not download_models():
        logger.error("Failed to download models")
        return 1
    
    # Check GPU support
    check_gpu_support()
    
    # Validate installation
    if not validate_installation():
        logger.error("Installation validation failed")
        return 1
    
    # Print system info
    print_system_info()
    
    logger.info("\n🎉 Setup completed successfully!")
    logger.info("You can now run the modern autonomous driving system.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 