# config.py
import os

# GGCNN Configuration
GGCNN_ROOT_PATH = "/mnt/nova_ssd/workspaces/isaac_ros-dev/src/ggcnn/ggcnn"  # CHANGE THIS to your actual path
MODEL_PATH = os.path.join(GGCNN_ROOT_PATH, "ggcnn_weights_cornell", "ggcnn_epoch_23_cornell")

# Device Configuration  
DEVICE = "cuda"  # or "cpu"

# Image Processing Parameters
CROP_PADDING = 20
INPUT_SIZE = (224, 224)
MIN_QUALITY_THRESHOLD = 0.05

# Depth Processing Parameters
DEPTH_CLIP_MIN = 0.1  # meters
DEPTH_CLIP_MAX = 2.5   # meters

