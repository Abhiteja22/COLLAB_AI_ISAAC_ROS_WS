# grasp_processor.py
import numpy as np
from .ggcnn_predictor import GGCNNGraspPredictor

def extract_bbox_info(langsam_output):
    """Extract bounding boxes and labels from LangSAM output"""
    all_detections = []
    
    for detection in langsam_output:
        boxes = detection['boxes']
        labels = detection['labels']
        scores = detection['scores']
        
        for box, label, score in zip(boxes, labels, scores):
            all_detections.append({
                'bbox': box,
                'label': label,
                'score': score
            })
    
    return all_detections

def process_langsam_with_ggcnn(langsam_output, depth_image, model_path=None):
    """
    Complete pipeline: LangSAM output -> GGCNN grasp prediction
    
    Args:
        langsam_output: Your LangSAM detection results
        depth_image: Corresponding depth image (numpy array)
        model_path: Optional custom model path
    
    Returns:
        grasp_results: List of grasp predictions for each detected object
    """
    # Initialize GGCNN predictor
    predictor = GGCNNGraspPredictor(model_path) if model_path else GGCNNGraspPredictor()
    
    all_grasp_results = []
    
    # Extract detection information
    detections = extract_bbox_info(langsam_output)
    
    print(f"Processing {len(detections)} detected objects...")
    
    for i, detection in enumerate(detections):
        print(f"\nProcessing object {i+1}: {detection['label']} (score: {detection['score']:.3f})")
        
        try:
            # Step 1: Crop depth image
            cropped_depth, crop_coords = predictor.crop_depth_with_bbox(depth_image, detection['bbox'])
            print(f"  Cropped region size: {cropped_depth.shape}")
            
            # Step 2: Preprocess for GGCNN
            depth_tensor = predictor.preprocess_depth_image(cropped_depth)
            print(f"  Preprocessed tensor shape: {depth_tensor.shape}")
            
            # Step 3: Run GGCNN inference
            grasp_predictions = predictor.predict_grasp(depth_tensor)
            print(f"  GGCNN inference completed")
            
            # Step 4: Find best grasp
            best_grasp = predictor.get_best_grasp(grasp_predictions, crop_coords)
            
            if best_grasp is not None:
                result = {
                    'object_id': i,
                    'object_label': detection['label'],
                    'detection_score': detection['score'],
                    'bbox': detection['bbox'],
                    'grasp_pose': best_grasp
                }
                
                all_grasp_results.append(result)
                
                print(f"  ✓ Best grasp found:")
                print(f"    Position: ({best_grasp['x']:.1f}, {best_grasp['y']:.1f})")
                print(f"    Angle: {best_grasp['angle']:.3f} rad ({np.degrees(best_grasp['angle']):.1f}°)")
                print(f"    Width: {best_grasp['width']:.3f}")
                print(f"    Quality: {best_grasp['quality']:.3f}")
            else:
                print(f"  ✗ No valid grasp found for {detection['label']}")
                
        except Exception as e:
            print(f"  ✗ Error processing {detection['label']}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    return all_grasp_results
