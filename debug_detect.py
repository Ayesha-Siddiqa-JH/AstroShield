#!/usr/bin/env python3
"""
Debug script for YOLO coordinate detection issues.
Tests the coordinate pipeline and outputs debug images with drawn boxes.
"""
import argparse
import json
import os
import sys
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def draw_boxes_on_image(image_path, boxes, labels, scores, output_path, color=(0, 255, 0), thickness=3):
    """
    Draw bounding boxes on an image using OpenCV.
    
    Args:
        image_path: Path to input image
        boxes: List of boxes in format [(x1, y1, x2, y2), ...]
        labels: List of label strings
        scores: List of confidence scores
        output_path: Path to save output image
        color: BGR color tuple for boxes
        thickness: Box line thickness
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    img_h, img_w = img.shape[:2]
    print(f"📸 Image dimensions: {img_w}x{img_h}")
    
    # Draw each box
    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        x1, y1, x2, y2 = box
        
        # Verify coordinates are within bounds
        if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h:
            print(f"⚠️ Box {i} coordinates out of bounds: ({x1:.1f}, {y1:.1f}) -> ({x2:.1f}, {y2:.1f})")
            print(f"   Image size: {img_w}x{img_h}")
        
        # Clamp to image bounds
        x1 = max(0, min(img_w, int(x1)))
        y1 = max(0, min(img_h, int(y1)))
        x2 = max(0, min(img_w, int(x2)))
        y2 = max(0, min(img_h, int(y2)))
        
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label
        label_text = f"{label} {score:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        # Label background
        cv2.rectangle(
            img,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            color,
            -1
        )
        
        # Label text
        cv2.putText(
            img,
            label_text,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        
        print(f"✅ Box {i}: {label} at ({x1}, {y1}) -> ({x2}, {y2})")
    
    cv2.imwrite(output_path, img)
    print(f"💾 Saved debug image to: {output_path}")
    return img

def debug_detect(image_path, model_path, output_dir, conf_threshold=0.5):
    """
    Run YOLO detection and debug coordinate pipeline.
    
    Args:
        image_path: Path to input image
        model_path: Path to YOLO model weights
        output_dir: Directory to save debug outputs
        conf_threshold: Confidence threshold
    """
    print("=" * 70)
    print("🔍 YOLO Coordinate Detection Debug")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read original image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    orig_height, orig_width = img.shape[:2]
    print(f"\n📐 Original Image Dimensions: {orig_width}x{orig_height}")
    
    # Load model
    if not os.path.exists(model_path):
        print(f"⚠️ Model not found at {model_path}, using placeholder")
        # Create placeholder output
        debug_data = {
            "image_path": image_path,
            "original_size": {"width": orig_width, "height": orig_height},
            "model_path": model_path,
            "model_loaded": False,
            "detections": [],
            "error": "Model not found"
        }
        with open(os.path.join(output_dir, "debug_output.json"), "w") as f:
            json.dump(debug_data, f, indent=2)
        return
    
    print(f"🤖 Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Run inference
    print(f"\n🔬 Running inference with imgsz=640, conf={conf_threshold}...")
    results = model.predict(
        source=image_path,
        imgsz=640,
        conf=conf_threshold,
        iou=0.45,
        max_det=300,
        verbose=False
    )
    
    if not results or len(results) == 0:
        print("❌ No results returned from model")
        return
    
    result = results[0]
    
    # Get image dimensions from YOLO
    yolo_orig_shape = None
    yolo_shape = None
    
    if hasattr(result, 'orig_shape') and result.orig_shape is not None:
        yolo_orig_shape = result.orig_shape  # (height, width)
        print(f"📐 YOLO orig_shape: {yolo_orig_shape[1]}x{yolo_orig_shape[0]} (width x height)")
    
    if hasattr(result, 'shape') and result.shape is not None:
        yolo_shape = result.shape  # (height, width) of processed image
        print(f"📐 YOLO shape (processed): {yolo_shape[1]}x{yolo_shape[0]} (width x height)")
    
    # Verify dimensions match
    if yolo_orig_shape:
        yolo_w, yolo_h = yolo_orig_shape[1], yolo_orig_shape[0]
        if yolo_w != orig_width or yolo_h != orig_height:
            print(f"⚠️ DIMENSION MISMATCH!")
            print(f"   File: {orig_width}x{orig_height}")
            print(f"   YOLO: {yolo_w}x{yolo_h}")
        else:
            print(f"✅ Dimensions match: {orig_width}x{orig_height}")
    
    # Extract detections
    detections = []
    boxes = []
    labels = []
    scores = []
    
    if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
        print(f"\n📦 Found {len(result.boxes)} detections")
        
        for i, box in enumerate(result.boxes):
            # Get box data
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            label = result.names[class_id] if hasattr(result, 'names') else f"Class_{class_id}"
            
            # Get coordinates - box.xyxy returns [x1, y1, x2, y2] in original image space
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            print(f"\n🔍 Detection {i}:")
            print(f"   Label: {label}")
            print(f"   Confidence: {confidence:.4f}")
            print(f"   Raw coords from YOLO: ({x1:.2f}, {y1:.2f}) -> ({x2:.2f}, {y2:.2f})")
            
            # Verify coordinates are within bounds
            if x1 < 0 or y1 < 0 or x2 > orig_width or y2 > orig_height:
                print(f"   ⚠️ Coordinates exceed image bounds!")
                print(f"      Image: {orig_width}x{orig_height}")
                print(f"      Coords: ({x1:.2f}, {y1:.2f}) -> ({x2:.2f}, {y2:.2f})")
            
            # Clamp coordinates
            x1_clamped = max(0, min(orig_width, float(x1)))
            y1_clamped = max(0, min(orig_height, float(y1)))
            x2_clamped = max(0, min(orig_width, float(x2)))
            y2_clamped = max(0, min(orig_height, float(y2)))
            
            if x1 != x1_clamped or y1 != y1_clamped or x2 != x2_clamped or y2 != y2_clamped:
                print(f"   ⚠️ Coordinates clamped:")
                print(f"      Before: ({x1:.2f}, {y1:.2f}) -> ({x2:.2f}, {y2:.2f})")
                print(f"      After:  ({x1_clamped:.2f}, {y1_clamped:.2f}) -> ({x2_clamped:.2f}, {y2_clamped:.2f})")
            
            # Calculate box center and size
            center_x = (x1_clamped + x2_clamped) / 2
            center_y = (y1_clamped + y2_clamped) / 2
            box_width = x2_clamped - x1_clamped
            box_height = y2_clamped - y1_clamped
            
            print(f"   Final coords: ({x1_clamped:.2f}, {y1_clamped:.2f}) -> ({x2_clamped:.2f}, {y2_clamped:.2f})")
            print(f"   Box center: ({center_x:.2f}, {center_y:.2f})")
            print(f"   Box size: {box_width:.2f}x{box_height:.2f}")
            
            detections.append({
                "label": label,
                "score": confidence,
                "x1": float(x1_clamped),
                "y1": float(y1_clamped),
                "x2": float(x2_clamped),
                "y2": float(y2_clamped),
                "center_x": float(center_x),
                "center_y": float(center_y),
                "width": float(box_width),
                "height": float(box_height)
            })
            
            boxes.append((x1_clamped, y1_clamped, x2_clamped, y2_clamped))
            labels.append(label)
            scores.append(confidence)
    else:
        print("⚠️ No detections found")
    
    # Save debug JSON
    debug_data = {
        "image_path": image_path,
        "original_size": {
            "width": orig_width,
            "height": orig_height
        },
        "yolo_orig_shape": list(yolo_orig_shape) if yolo_orig_shape else None,
        "yolo_shape": list(yolo_shape) if yolo_shape else None,
        "model_path": model_path,
        "model_loaded": True,
        "conf_threshold": conf_threshold,
        "detections": detections,
        "coordinate_system": "original_image_space_xyxy",
        "notes": "Coordinates from box.xyxy are in original image pixel space"
    }
    
    json_path = os.path.join(output_dir, "debug_output.json")
    with open(json_path, "w") as f:
        json.dump(debug_data, f, indent=2)
    print(f"\n💾 Saved debug JSON to: {json_path}")
    
    # Draw boxes on image
    if boxes:
        output_image_path = os.path.join(output_dir, "debug_output.jpg")
        draw_boxes_on_image(image_path, boxes, labels, scores, output_image_path)
        print(f"\n✅ Debug complete! Check:")
        print(f"   - Image: {output_image_path}")
        print(f"   - JSON: {json_path}")
    else:
        print("\n⚠️ No boxes to draw")
    
    print("\n" + "=" * 70)

def main():
    parser = argparse.ArgumentParser(description="Debug YOLO coordinate detection")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="models/weights/best.pt",
        help="Path to model weights"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="debug_out",
        help="Output directory for debug files"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"❌ Image not found: {args.image}")
        sys.exit(1)
    
    debug_detect(args.image, args.weights, args.out, args.conf)

if __name__ == "__main__":
    main()

