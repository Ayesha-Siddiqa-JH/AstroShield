#!/usr/bin/env python3
"""Test detect function directly."""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.detect import detect

MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'weights', 'best.pt')
TEST_IMAGE = os.path.join(PROJECT_ROOT, 'data', 'raw', 'test', 'images', '000000000_vcluttered_hallway.png')

print(f"Model path: {MODEL_PATH}")
print(f"Model exists: {os.path.exists(MODEL_PATH)}")
print(f"Test image: {TEST_IMAGE}")
print(f"Test image exists: {os.path.exists(TEST_IMAGE)}")

if os.path.exists(TEST_IMAGE):
    try:
        print("\nRunning detection...")
        results = detect(TEST_IMAGE, model_path=MODEL_PATH)
        print(f"✅ Detection successful!")
        print(f"Results type: {type(results)}")
        if results and len(results) > 0:
            result = results[0]
            print(f"Number of detections: {len(result.boxes)}")
            for i, box in enumerate(result.boxes):
                class_id = int(box.cls.item())
                label = result.names[class_id]
                confidence = float(box.conf.item())
                print(f"  {i+1}. {label}: {confidence:.2f}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

