# test_integration.py
import numpy as np
import cv2
from .grasp_processor import process_langsam_with_ggcnn

def main():
    """Test the LangSAM + GGCNN integration"""
    
    # Your actual LangSAM output
    # Car
    # langsam_output = [{
    #     'scores': np.array([0.8367713, 0.83055925], dtype=np.float32),
    #     'labels': ['wheel', 'wheel'],
    #     'boxes': np.array([[460.7055, 228.73825, 558.312, 322.39478],
    #                       [117.74089, 229.1464, 213.98326, 323.87173]], dtype=np.float32),
    #     'masks': np.random.rand(2, 450, 680).astype(np.float32),  # Replace with actual masks
    #     'mask_scores': np.array([0.97150326, 0.9681552], dtype=np.float32)
    # }]

    # Fruits
    langsam_output = [{
        'scores': np.array([0.35344404, 0.3444759 , 0.37005422, 0.30139458], dtype=np.float32), 
        'labels': ['strawberry', 'strawberry', 'strawberry', 'strawberry'], 
        'boxes': np.array([[226.51028, 393.82562, 314.58923, 481.4544 ],
                        [340.99277, 383.18506, 453.94305, 467.6969 ],
                        [337.6512 , 258.9217 , 430.52557, 372.76108],
                        [273.6812 , 337.5617 , 344.54135, 417.60944]], dtype=np.float32), 
        'masks': np.random.rand(2, 450, 680).astype(np.float32), 
        'mask_scores': np.array([0.9439467 , 0.9368757 , 0.94181526, 0.93539226], dtype=np.float32)}]
    
    # Load your depth image - REPLACE THIS PATH
    depth_image_path = "path/to/your/depth_image.png"  # CHANGE THIS
    
    try:
        # Try loading depth image
        depth_image = cv2.imread(depth_image_path, cv2.IMREAD_ANYDEPTH)
        if depth_image is None:
            print(f"Warning: Could not load depth image from {depth_image_path}")
            print("Creating dummy depth image for testing...")
            # Create dummy depth image with same dimensions as LangSAM expects
            depth_image = np.random.randint(100, 2000, (450, 680), dtype=np.uint16)
        
        print(f"Depth image shape: {depth_image.shape}")
        print(f"Depth image dtype: {depth_image.dtype}")
        print(f"Depth range: {depth_image.min()} - {depth_image.max()}")
        
        # Process with GGCNN
        grasp_results = process_langsam_with_ggcnn(langsam_output, depth_image)
        
        # Print final results
        print(f"\n{'='*50}")
        print(f"FINAL RESULTS: Found {len(grasp_results)} grasp poses")
        print(f"{'='*50}")
        
        for i, result in enumerate(grasp_results):
            print(f"\nObject {i+1}: {result['object_label']}")
            print(f"  Detection confidence: {result['detection_score']:.3f}")
            pose = result['grasp_pose']
            print(f"  Grasp position: ({pose['x']:.1f}, {pose['y']:.1f}) pixels")
            print(f"  Grasp angle: {pose['angle']:.3f} rad ({np.degrees(pose['angle']):.1f}°)")
            print(f"  Grasp width: {pose['width']:.3f} meters")
            print(f"  Grasp quality: {pose['quality']:.3f}")
            
            # These coordinates can now be used for robot control!
            print(f"  → Ready for robot control at pixel ({pose['x']:.0f}, {pose['y']:.0f})")
        
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
