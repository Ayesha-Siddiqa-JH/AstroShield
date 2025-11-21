import os, cv2
from ultralytics import YOLO

def detect(image_path, model_path='../best.pt', save_dir=None, conf_threshold=0.3):
    """
    Run YOLO detection on an image.
    
    Args:
        image_path: Path to input image
        model_path: Path to YOLO model weights
        save_dir: Directory to save detection results
        conf_threshold: Confidence threshold (0.0-1.0), default 0.25
    
    Returns:
        Detection results (coordinates are in original image space)
    """
    # Default save directory if not provided
    if save_dir is None:
        save_dir = os.environ.get('SAVE_DIR', 'models/logs/detect')
    
    # Make sure directories exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Load the model
    model = YOLO(model_path)
    
    # Read original image to get dimensions
    img = cv2.imread(image_path)
    if img is not None:
        orig_height, orig_width = img.shape[:2]
        print(f"Original image size: {orig_width}x{orig_height}")
    
    # Run inference with confidence threshold
    # Note: YOLO automatically handles letterbox padding and returns coordinates in original image space
    results = model.predict(
        source=image_path, 
        save=True, 
        save_dir=save_dir, 
        imgsz=640,  # Model input size (YOLO handles resizing with letterbox)
        conf=conf_threshold,  # Confidence threshold
        iou=0.45,  # NMS IoU threshold
        max_det=300,  # Maximum detections per image
        verbose=False  # Reduce output
    )
    
    # Verify coordinates are in original image space
    if results and len(results) > 0:
        result = results[0]
        if hasattr(result, 'orig_shape') and result.orig_shape:
            print(f"YOLO reports original shape: {result.orig_shape[1]}x{result.orig_shape[0]}")
        if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
            # Check first box coordinates
            first_box = result.boxes[0]
            x1, y1, x2, y2 = first_box.xyxy[0].tolist()
            print(f"First detection coordinates: ({x1:.1f}, {y1:.1f}) -> ({x2:.1f}, {y2:.1f})")
    
    print(f"Detections saved to {save_dir}")
    return results

if __name__ == '__main__':
    import sys
    img = sys.argv[1] if len(sys.argv)>1 else '../data/raw/test/images/sample.jpg'
    detect(img)