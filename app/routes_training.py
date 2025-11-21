"""
Training Routes for Interactive Model Training
Allows users to upload images and label them for training
"""
from flask import Blueprint, request, jsonify
import os
import sys
import logging
import shutil
from datetime import datetime

logger = logging.getLogger(__name__)

training_routes = Blueprint('training', __name__)

# Training data directories
TRAIN_IMAGES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw', 'train', 'images')
TRAIN_LABELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw', 'train', 'labels')
CUSTOM_TRAIN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'custom_training')

# Ensure directories exist
os.makedirs(TRAIN_IMAGES_DIR, exist_ok=True)
os.makedirs(TRAIN_LABELS_DIR, exist_ok=True)
os.makedirs(CUSTOM_TRAIN_DIR, exist_ok=True)

# Class mapping
CLASS_MAP = {
    'OxygenTank': 0,
    'NitrogenTank': 1,
    'FirstAidBox': 2,
    'FireAlarm': 3,
    'SafetySwitchPanel': 4,
    'EmergencyPhone': 5,
    'FireExtinguisher': 6
}

@training_routes.route('/api/train/upload', methods=['POST'])
def upload_training_image():
    """Upload an image for training. Accepts either single label or multiple annotations."""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Save image first - use original filename or custom name if provided
        # Keep original filename structure but allow custom naming
        original_filename = file.filename
        if not original_filename:
            original_filename = 'image.jpg'
        
        # Use custom name if provided in form, otherwise use timestamp + original
        custom_name = request.form.get('image_name', '').strip()
        if custom_name:
            # Add extension if not present
            if '.' not in custom_name:
                ext = original_filename.rsplit('.', 1)[-1] if '.' in original_filename else 'jpg'
                custom_name = f"{custom_name}.{ext}"
            filename = custom_name
        else:
            filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{original_filename}"
        
        image_path = os.path.join(TRAIN_IMAGES_DIR, filename)
        file.save(image_path)
        
        # Get image dimensions for coordinate conversion
        from PIL import Image
        img = Image.open(image_path)
        img_width, img_height = img.size
        
        # Create label file (YOLO format: class_id center_x center_y width height)
        label_filename = filename.rsplit('.', 1)[0] + '.txt'
        label_path = os.path.join(TRAIN_LABELS_DIR, label_filename)
        
        # Check if annotations are provided (multiple objects)
        if 'annotations' in request.form:
            # Multiple bounding boxes provided
            import json
            annotations = json.loads(request.form['annotations'])
            
            with open(label_path, 'w') as f:
                for ann in annotations:
                    label = ann.get('label', '').strip()
                    if label not in CLASS_MAP:
                        continue
                    
                    class_id = CLASS_MAP[label]
                    # Convert pixel coordinates to YOLO format (normalized 0-1)
                    x1 = ann.get('x1', 0)
                    y1 = ann.get('y1', 0)
                    x2 = ann.get('x2', img_width)
                    y2 = ann.get('y2', img_height)
                    
                    # Calculate center and dimensions (normalized)
                    center_x = ((x1 + x2) / 2) / img_width
                    center_y = ((y1 + y2) / 2) / img_height
                    width = abs(x2 - x1) / img_width
                    height = abs(y2 - y1) / img_height
                    
                    # Ensure values are within [0, 1]
                    center_x = max(0, min(1, center_x))
                    center_y = max(0, min(1, center_y))
                    width = max(0, min(1, width))
                    height = max(0, min(1, height))
                    
                    f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
            
            return jsonify({
                'success': True,
                'message': f'Image uploaded with {len(annotations)} annotations',
                'image_path': image_path,
                'label_path': label_path,
                'annotations_count': len(annotations)
            })
        
        # Legacy: Single label (for backward compatibility)
        elif 'label' in request.form:
            label = request.form['label'].strip()
            
            if label not in CLASS_MAP:
                return jsonify({
                    'error': f'Invalid label. Must be one of: {", ".join(CLASS_MAP.keys())}'
                }), 400
            
            class_id = CLASS_MAP[label]
            # Placeholder: full image bounding box
            with open(label_path, 'w') as f:
                f.write(f"{class_id} 0.5 0.5 1.0 1.0")
            
            return jsonify({
                'success': True,
                'message': f'Image uploaded and labeled as {label}',
                'image_path': image_path,
                'label_path': label_path,
                'class_id': class_id
            })
        else:
            return jsonify({'error': 'No label or annotations provided'}), 400
        
    except Exception as e:
        logger.error(f"Error uploading training image: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@training_routes.route('/api/train/status', methods=['GET'])
def training_status():
    """Get training data statistics."""
    try:
        image_count = len([f for f in os.listdir(TRAIN_IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        label_count = len([f for f in os.listdir(TRAIN_LABELS_DIR) if f.endswith('.txt')])
        
        # Count by class (count all annotations, not just files)
        class_counts = {name: 0 for name in CLASS_MAP.keys()}
        total_annotations = 0
        for label_file in os.listdir(TRAIN_LABELS_DIR):
            if label_file.endswith('.txt'):
                label_path = os.path.join(TRAIN_LABELS_DIR, label_file)
                try:
                    with open(label_path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                parts = line.split()
                                if len(parts) >= 5:
                                    class_id = int(parts[0])
                                    total_annotations += 1
                                    # Find class name by ID
                                    for name, id_val in CLASS_MAP.items():
                                        if id_val == class_id:
                                            class_counts[name] += 1
                                            break
                except Exception as e:
                    logger.warning(f"Error reading label file {label_file}: {e}")
                    pass
        
        return jsonify({
            'success': True,
            'total_images': image_count,
            'total_labels': label_count,
            'total_annotations': total_annotations,
            'class_counts': class_counts,
            'classes': list(CLASS_MAP.keys())
        })
        
    except Exception as e:
        logger.error(f"Error getting training status: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@training_routes.route('/api/train/retrain', methods=['POST'])
def trigger_retrain():
    """Trigger model retraining with current training data."""
    try:
        # Check if training data exists
        image_count = len([f for f in os.listdir(TRAIN_IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        label_count = len([f for f in os.listdir(TRAIN_LABELS_DIR) if f.endswith('.txt')])
        
        if image_count < 10:
            return jsonify({
                'error': f'Not enough training data. Need at least 10 images, have {image_count}'
            }), 400
        
        if image_count != label_count:
            return jsonify({
                'error': f'Mismatch: {image_count} images but {label_count} labels. Each image needs a label file.'
            }), 400
        
        # Check current model
        try:
            from ultralytics import YOLO
        except ImportError:
            return jsonify({
                'success': False,
                'error': 'ultralytics module not installed. Please install: pip install ultralytics'
            }), 503
        
        current_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'weights', 'best.pt')
        if not os.path.exists(current_model_path):
            return jsonify({
                'success': False,
                'error': f'Model not found at {current_model_path}. Please train a model first.'
            }), 404
        
        current_model = YOLO(current_model_path)
        current_classes = len(current_model.names)
        
        return jsonify({
            'success': True,
            'message': 'Ready to retrain! Run: python retrain_model.py',
            'training_images': image_count,
            'label_files': label_count,
            'current_model_classes': current_classes,
            'expected_classes': 7,
            'note': 'Retraining takes 2-8 hours. This will create a new model that knows all 7 classes including your new training data.',
            'instructions': [
                '1. Open a new terminal/PowerShell window',
                '2. Navigate to project: cd "D:\\My Downloads\\AstroShield_project"',
                '3. Activate environment: conda activate observo',
                '4. Start training: python retrain_model.py',
                '5. Wait for training to complete (2-8 hours)',
                '6. Restart server: python start_server.py',
                '7. Test detection - it will work correctly!'
            ]
        })
        
    except Exception as e:
        logger.error(f"Error triggering retrain: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

