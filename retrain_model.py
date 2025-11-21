#!/usr/bin/env python3
"""
Retrain YOLO Model with Correct 7 Classes
=========================================
This script retrains the model with the proper 7-class configuration
to fix the FirstAidBox detection issue.
"""

import os
import sys
from pathlib import Path

def main():
    print("🚀 AstroShield Model Retraining")
    print("=" * 50)
    
    try:
        from ultralytics import YOLO
        import torch
        
        # Check device
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"🖥️  Using device: {device}")
        if device == 'cuda:0':
            print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        
        # Get project root
        project_root = Path(__file__).parent
        data_config = project_root / 'data' / 'raw' / 'yolo_params.yaml'
        
        if not data_config.exists():
            print(f"❌ Configuration file not found: {data_config}")
            print("   Creating it now...")
            # Create the config file
            config_content = """# YOLO Training Configuration for AstroShield
path: .
train: train/images
val: test/images
test: test/images

nc: 7
names:
  0: OxygenTank
  1: NitrogenTank
  2: FirstAidBox
  3: FireAlarm
  4: SafetySwitchPanel
  5: EmergencyPhone
  6: FireExtinguisher
"""
            data_config.parent.mkdir(parents=True, exist_ok=True)
            data_config.write_text(config_content)
            print(f"✅ Created {data_config}")
        
        print(f"📊 Using data config: {data_config}")
        
        # Verify training data exists
        train_images = project_root / 'data' / 'raw' / 'train' / 'images'
        train_labels = project_root / 'data' / 'raw' / 'train' / 'labels'
        
        if not train_images.exists() or not train_labels.exists():
            print(f"❌ Training data not found!")
            print(f"   Expected: {train_images}")
            print(f"   Expected: {train_labels}")
            return False
        
        train_count = len(list(train_images.glob('*.png')))
        label_count = len(list(train_labels.glob('*.txt')))
        print(f"📦 Training images: {train_count}")
        print(f"📦 Training labels: {label_count}")
        
        if train_count == 0 or label_count == 0:
            print("❌ No training data found!")
            return False
        
        # Load model
        print("\n📥 Loading YOLOv8n base model...")
        model = YOLO('yolov8n.pt')
        
        print("\n🎯 Training Configuration:")
        print("   📈 Epochs: 300")
        print("   🖼️  Image Size: 640x640")
        print("   📦 Batch Size: 8 (adjust if GPU memory is limited)")
        print("   🎨 Advanced Augmentation: Enabled")
        print("   ⚡ Optimized Hyperparameters: Applied")
        print("   🎯 Classes: 7 (OxygenTank, NitrogenTank, FirstAidBox, FireAlarm, SafetySwitchPanel, EmergencyPhone, FireExtinguisher)")
        
        print("\n🔥 Starting training...")
        print("   This may take several hours depending on your hardware.")
        print("   Training progress will be saved to: models/logs/yolov8_astroshield/")
        
        # Start training with optimized parameters for better mAP@0.5
        results = model.train(
            data=str(data_config),
            epochs=300,
            imgsz=640,
            batch=8,
            device=device,
            
            # Advanced augmentation
            mosaic=0.9,
            mixup=0.15,
            copy_paste=0.3,
            
            # Optimized hyperparameters
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.9,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            
            # Learning rate scheduling
            lr0=0.01,
            lrf=0.001,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            
            # Loss function weights for better mAP
            box=7.5,
            cls=0.5,
            dfl=1.5,
            
            # Validation settings
            val=True,
            save=True,
            save_period=10,
            project='models/logs',
            name='yolov8_astroshield',
            exist_ok=True,
            
            # Performance optimization
            workers=8,
            patience=50,  # Early stopping patience
            close_mosaic=10,  # Disable mosaic in last 10 epochs
        )
        
        print("\n✅ Training completed!")
        print(f"📊 Results saved to: {results.save_dir}")
        print(f"📁 Best model: {results.save_dir / 'weights' / 'best.pt'}")
        
        # Copy best model to models/weights/
        best_model = results.save_dir / 'weights' / 'best.pt'
        target_model = project_root / 'models' / 'weights' / 'best.pt'
        
        if best_model.exists():
            import shutil
            shutil.copy2(best_model, target_model)
            print(f"✅ Model copied to: {target_model}")
            print("\n🎉 Model retraining complete! The new model should now correctly detect all 7 classes.")
        else:
            print("⚠️ Best model not found, check training logs")
        
        return True
        
    except ImportError as e:
        print(f"❌ Missing required packages: {e}")
        print("   Install with: pip install ultralytics torch torchvision")
        return False
    except Exception as e:
        print(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

