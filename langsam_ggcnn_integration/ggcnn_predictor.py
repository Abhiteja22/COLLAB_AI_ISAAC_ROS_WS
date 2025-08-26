# ggcnn_predictor.py
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import sys
import os
import warnings

# Suppress the NumPy warning
warnings.filterwarnings("ignore", message="Failed to initialize NumPy")

# Import configuration
from .config import GGCNN_ROOT_PATH, MODEL_PATH, DEVICE, INPUT_SIZE, DEPTH_CLIP_MIN, DEPTH_CLIP_MAX

# Add GGCNN to Python path
sys.path.append(GGCNN_ROOT_PATH)

try:
    from models.common import post_process_output
    from models.ggcnn import GGCNN
    #from utils.data.camera_data import CameraData
except ImportError as e:
    print(f"Error importing GGCNN modules: {e}")
    print(f"Make sure GGCNN_ROOT_PATH in config.py points to: {GGCNN_ROOT_PATH}")
    
    # Try alternative import paths
    try:
        sys.path.append(os.path.join(GGCNN_ROOT_PATH, 'ggcnn'))
        from models.ggcnn import GGCNN
        from models.common import post_process_output
        print("Successfully imported with alternative path")
    except ImportError as e2:
        print(f"Alternative import also failed: {e2}")
        raise

class GGCNNGraspPredictor:
    def __init__(self, model_path=MODEL_PATH, device=DEVICE):
        """
        Initialize GGCNN grasp predictor
        Args:
            model_path: Path to trained GGCNN model weights
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.model = self.load_model(model_path)
        self.model.eval()
        print("GGCNN model loaded successfully!")

    def load_model(self, model_path):
        """Load the trained GGCNN model with robust error handling"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"Loading model from: {model_path}")
        
        try:
            # Load the checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            print(f"Checkpoint type: {type(checkpoint)}")
            
            # Initialize GGCNN model
            model = GGCNN(input_channels=1)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print("✓ Loaded from 'model_state_dict' key")
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                    print("✓ Loaded from 'state_dict' key")
                elif 'model' in checkpoint:
                    model.load_state_dict(checkpoint['model'])
                    print("✓ Loaded from 'model' key")
                else:
                    # Assume the entire dict is the state_dict
                    model.load_state_dict(checkpoint)
                    print("✓ Loaded from direct state_dict")
            elif hasattr(checkpoint, 'state_dict'):
                # Extract state_dict from model object
                model.load_state_dict(checkpoint.state_dict())
                print("✓ Loaded from model object's state_dict")
            else:
                # The checkpoint IS the model
                if isinstance(checkpoint, GGCNN):
                    model = checkpoint
                    print("✓ Using loaded model object directly")
                else:
                    raise ValueError(f"Unknown checkpoint format: {type(checkpoint)}")
            
            model.to(self.device)
            return model
            
        except Exception as e:
            print(f"Standard loading failed: {e}")
            return self._try_alternative_loading(model_path)

    def _try_alternative_loading(self, model_path):
        """Try alternative model loading methods"""
        print("Trying alternative loading methods...")
        
        # Method 1: Try loading with weights_only=False
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            if isinstance(checkpoint, GGCNN):
                print("✓ Successfully loaded complete model object")
                return checkpoint.to(self.device)
            elif hasattr(checkpoint, '__dict__') and hasattr(checkpoint, 'state_dict'):
                model = GGCNN(input_channels=1)
                model.load_state_dict(checkpoint.state_dict())
                print("✓ Alternative method 1 successful")
                return model.to(self.device)
                
        except Exception as e:
            print(f"Method 1 failed: {e}")
        
        # Method 2: Try with pickle protocol
        try:
            import pickle
            with open(model_path, 'rb') as f:
                checkpoint = pickle.load(f)
            
            if isinstance(checkpoint, GGCNN):
                print("✓ Loaded with pickle")
                return checkpoint.to(self.device)
                
        except Exception as e:
            print(f"Pickle method failed: {e}")
        
        raise RuntimeError("All model loading methods failed. Please check your model file format.")

    def preprocess_depth_image(self, depth_image):
        """
        Preprocess depth image for GGCNN input
        Args:
            depth_image: Raw depth image from camera
        Returns:
            Processed depth tensor
        """
        # Convert to numpy if PIL Image
        if isinstance(depth_image, Image.Image):
            depth_image = np.array(depth_image)
        
        # Handle different depth formats
        if depth_image.dtype == np.uint16:
            depth_image = depth_image.astype(np.float32) / 1000.0  # Convert mm to m
        elif depth_image.dtype == np.uint8:
            depth_image = depth_image.astype(np.float32) / 255.0
        
        # Remove invalid values
        depth_image = np.nan_to_num(depth_image, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Clip depth values
        depth_image = np.clip(depth_image, DEPTH_CLIP_MIN, DEPTH_CLIP_MAX)
        
        # Resize to GGCNN input size
        depth_image = cv2.resize(depth_image, INPUT_SIZE)
        
        # Normalize depth values
        if depth_image.std() > 0:
            depth_image = (depth_image - depth_image.mean()) / depth_image.std()
        
        # Convert to tensor
        depth_tensor = torch.from_numpy(depth_image).float()
        depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        
        return depth_tensor.to(self.device)

    def crop_depth_with_bbox(self, depth_image, bbox, padding=None):
        """Crop depth image using bounding box from LangSAM"""
        if padding is None:
            from .config import CROP_PADDING
            padding = CROP_PADDING
        
        x1, y1, x2, y2 = bbox.astype(int)
        
        # Add padding and ensure within bounds
        h, w = depth_image.shape[:2]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        # Crop the depth image
        cropped_depth = depth_image[y1:y2, x1:x2]
        
        return cropped_depth, (x1, y1, x2, y2)

    def predict_grasp(self, depth_tensor):
        """Predict grasp using GGCNN"""
        with torch.no_grad():
            # Forward pass through GGCNN
            pos_out, cos_out, sin_out, width_out = self.model(depth_tensor)
            
            # Convert to numpy
            pos_pred = pos_out.cpu().numpy().squeeze()
            cos_pred = cos_out.cpu().numpy().squeeze()
            sin_pred = sin_out.cpu().numpy().squeeze()
            width_pred = width_out.cpu().numpy().squeeze()
            
            # Calculate angle from cos and sin
            angle_pred = np.arctan2(sin_pred, cos_pred) / 2.0
            
            return {
                'quality': pos_pred,
                'angle': angle_pred,
                'width': width_pred
            }

    def get_best_grasp(self, grasp_predictions, original_crop_coords=None, min_quality=None):
        """Extract the best grasp from predictions"""
        if min_quality is None:
            from .config import MIN_QUALITY_THRESHOLD
            min_quality = MIN_QUALITY_THRESHOLD
        
        quality_pred = grasp_predictions['quality']
        angle_pred = grasp_predictions['angle']
        width_pred = grasp_predictions['width']
        
        # Find valid grasps above threshold
        valid_mask = quality_pred > min_quality
        if not np.any(valid_mask):
            print(f"No grasps found above quality threshold {min_quality}")
            return None
        
        # Find best grasp location
        max_idx = np.unravel_index(np.argmax(quality_pred * valid_mask), quality_pred.shape)
        best_y, best_x = max_idx
        
        # Extract grasp parameters
        best_angle = angle_pred[best_y, best_x]
        best_width = width_pred[best_y, best_x]
        best_quality = quality_pred[best_y, best_x]
        
        # Convert back to original image coordinates if crop coords provided
        if original_crop_coords:
            x1, y1, x2, y2 = original_crop_coords
            
            # Scale from 224x224 back to crop size
            scale_x = (x2 - x1) / INPUT_SIZE[0]
            scale_y = (y2 - y1) / INPUT_SIZE[1]
            
            # Convert to original image coordinates
            original_x = x1 + best_x * scale_x
            original_y = y1 + best_y * scale_y
        else:
            original_x = best_x
            original_y = best_y
        
        return {
            'x': float(original_x),
            'y': float(original_y),
            'angle': float(best_angle),
            'width': float(best_width),
            'quality': float(best_quality)
        }
