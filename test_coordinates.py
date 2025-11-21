#!/usr/bin/env python3
"""
Unit test to verify coordinate transformations are correct.
Tests that box centers match expected positions.
"""
import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path

def test_coordinate_bounds(image_path, detections_json_path):
    """
    Test that all detection coordinates are within image bounds.
    """
    print("=" * 70)
    print("🧪 Coordinate Bounds Test")
    print("=" * 70)
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not read image: {image_path}")
        return False
    
    img_h, img_w = img.shape[:2]
    print(f"📐 Image size: {img_w}x{img_h}")
    
    # Read detections
    with open(detections_json_path, 'r') as f:
        data = json.load(f)
    
    detections = data.get('detections', [])
    print(f"📦 Found {len(detections)} detections")
    
    all_valid = True
    
    for i, det in enumerate(detections):
        x1, y1 = det['x1'], det['y1']
        x2, y2 = det['x2'], det['y2']
        
        # Check bounds
        valid = True
        if x1 < 0 or x1 > img_w:
            print(f"❌ Detection {i}: x1={x1:.2f} out of bounds [0, {img_w}]")
            valid = False
        if y1 < 0 or y1 > img_h:
            print(f"❌ Detection {i}: y1={y1:.2f} out of bounds [0, {img_h}]")
            valid = False
        if x2 < 0 or x2 > img_w:
            print(f"❌ Detection {i}: x2={x2:.2f} out of bounds [0, {img_w}]")
            valid = False
        if y2 < 0 or y2 > img_h:
            print(f"❌ Detection {i}: y2={y2:.2f} out of bounds [0, {img_h}]")
            valid = False
        
        if not valid:
            all_valid = False
        else:
            print(f"✅ Detection {i}: {det['label']} - coords valid")
    
    if all_valid:
        print("\n✅ All coordinates are within image bounds!")
    else:
        print("\n❌ Some coordinates are out of bounds!")
    
    return all_valid

def test_coordinate_consistency(image_path, detections_json_path):
    """
    Test that coordinates are consistent (x2 > x1, y2 > y1, reasonable box sizes).
    """
    print("\n" + "=" * 70)
    print("🧪 Coordinate Consistency Test")
    print("=" * 70)
    
    with open(detections_json_path, 'r') as f:
        data = json.load(f)
    
    detections = data.get('detections', [])
    all_valid = True
    
    for i, det in enumerate(detections):
        x1, y1 = det['x1'], det['y1']
        x2, y2 = det['x2'], det['y2']
        width = det.get('width', x2 - x1)
        height = det.get('height', y2 - y1)
        
        # Check x2 > x1
        if x2 <= x1:
            print(f"❌ Detection {i}: x2 ({x2:.2f}) <= x1 ({x1:.2f})")
            all_valid = False
        
        # Check y2 > y1
        if y2 <= y1:
            print(f"❌ Detection {i}: y2 ({y2:.2f}) <= y1 ({y1:.2f})")
            all_valid = False
        
        # Check reasonable box size (at least 10x10 pixels)
        if width < 10 or height < 10:
            print(f"⚠️ Detection {i}: Very small box {width:.1f}x{height:.1f}")
        
        # Check center matches
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        expected_center_x = det.get('center_x', center_x)
        expected_center_y = det.get('center_y', center_y)
        
        if abs(center_x - expected_center_x) > 0.1 or abs(center_y - expected_center_y) > 0.1:
            print(f"⚠️ Detection {i}: Center mismatch")
            print(f"   Calculated: ({center_x:.2f}, {center_y:.2f})")
            print(f"   Expected: ({expected_center_x:.2f}, {expected_center_y:.2f})")
        
        if all_valid:
            print(f"✅ Detection {i}: {det['label']} - coords consistent")
    
    if all_valid:
        print("\n✅ All coordinates are consistent!")
    else:
        print("\n❌ Some coordinates are inconsistent!")
    
    return all_valid

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test coordinate transformations")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--json", type=str, required=True, help="Detections JSON path")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"❌ Image not found: {args.image}")
        sys.exit(1)
    
    if not os.path.exists(args.json):
        print(f"❌ JSON not found: {args.json}")
        sys.exit(1)
    
    test1 = test_coordinate_bounds(args.image, args.json)
    test2 = test_coordinate_consistency(args.image, args.json)
    
    if test1 and test2:
        print("\n" + "=" * 70)
        print("✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n" + "=" * 70)
        print("❌ Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()

