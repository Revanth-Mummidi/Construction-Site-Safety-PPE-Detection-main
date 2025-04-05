import os
from pathlib import Path

# Get the project root directory (two levels up from config/)
PROJECT_ROOT = Path(__file__).parent.parent

# Model Configuration
MODEL_PATH = str(PROJECT_ROOT / "models" / "best.pt")  # Absolute path to model
CONFIDENCE_THRESHOLD = 0.5
IMG_SIZE = 640

# Processing Configuration
TARGET_FPS = 24
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
IOU_THRESHOLD = 0

# PPE Weightage (must sum to 100%)
PPE_WEIGHTS = {
    'Hardhat': 40,
    'Safety Vest': 40,
    'Mask': 20
}

# Directories
SOURCE_DIR = str(PROJECT_ROOT / "source_files")
OUTPUT_DIR = str(PROJECT_ROOT / "outputs")
LOG_DIR = str(PROJECT_ROOT / "outputs" / "logs")

# Visualization
COLORS = {
    'Person': (0, 0, 255),        # Red
    'Hardhat': (0, 255, 0),       # Green
    'Safety Vest': (255, 255, 0), # Yellow
    'Mask': (0, 255, 255),        # Cyan
    'compliant': (0, 200, 0),     # Dark Green
    'non_compliant': (0, 0, 200)  # Dark Red
}

# Compliance Threshold
COMPLIANCE_THRESHOLD = 80  # Minimum score to be considered compliant