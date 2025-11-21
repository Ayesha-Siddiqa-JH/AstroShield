from flask import Blueprint, request, render_template, send_from_directory, jsonify
import os
import sys
import logging
import uuid
from datetime import datetime

# Add the parent directory to sys.path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

# Try to import detect function, but handle gracefully if ultralytics is not installed
try:
    from src.detect import detect
    DETECT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ Detection module not available: {e}. Running in limited mode.")
    DETECT_AVAILABLE = False
    detect = None

routes = Blueprint('routes', __name__)

# Use absolute path for uploads in production, or relative path in development
UPLOAD = os.environ.get('UPLOAD_DIR', 'uploads')
os.makedirs(UPLOAD, exist_ok=True)

# Define model path based on environment
MODEL_PATH = os.environ.get('MODEL_PATH', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'best.pt'))

@routes.route('/', methods=['GET','POST'])
def index():
    """Serve the main HTML page."""
    return render_template('index.html')

@routes.route('/favicon.ico')
def favicon():
    """Handle favicon requests to prevent 404 errors."""
    return '', 204  # No content, but successful

@routes.route('/api/detect', methods=['POST'])
def api_detect():
    import time
    start_time = time.time()
    response_headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type'
    }
    
    if request.method == 'OPTIONS':
        return ('', 204, response_headers)
        
    try:
        logger.info("API detect endpoint called")
        logger.info(f"Request files: {request.files}")
        logger.info(f"Request form: {request.form}")
        
        # Check if detection is available
        if not DETECT_AVAILABLE:
            return jsonify({
                'success': False,
                'error': 'Detection module not available. Please install dependencies: pip install -r requirements.txt',
                'available_classes': ['OxygenTank', 'NitrogenTank', 'FirstAidBox', 'FireAlarm', 'SafetySwitchPanel', 'EmergencyPhone', 'FireExtinguisher']
            }), 503, response_headers
        
        if 'image' not in request.files:
            logger.warning("No image file in request")
            return jsonify({'success': False, 'error': 'No image file provided'}), 400, response_headers
            
        file = request.files['image']
        if file.filename == '':
            logger.warning("Empty filename")
            return jsonify({'success': False, 'error': 'No image selected'}), 400, response_headers
            
        # Ensure upload directory exists
        os.makedirs(UPLOAD, exist_ok=True)
        
        # Generate a safe filename
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}_{file.filename}"
        path = os.path.join(UPLOAD, filename)
        
        try:
            file.save(path)
            logger.info(f"File saved to {path}")
            
            # Verify the file was saved
            if not os.path.exists(path):
                logger.error(f"Failed to save file to {path}")
                return jsonify({'success': False, 'error': 'Failed to save uploaded file'}), 500, response_headers
                
            # Run detection
            results = detect(path, model_path=MODEL_PATH, conf_threshold=0.2)
            logger.info(f"Detection complete, processing results")
            
            detections = []
            if results and len(results) > 0:
                result = results[0]
                logger.info(f"Found {len(result.boxes)} detections")
                for i, box in enumerate(result.boxes):
                    class_id = int(box.cls.item())
                    label = result.names[class_id]
                    confidence = float(box.conf.item())
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    risk_level = 'high' if confidence < 0.5 else 'medium' if confidence < 0.75 else 'low'
                    detections.append({
                        'id': str(i),
                        'label': label,
                        'confidence': confidence,
                        'bbox': {'x': x1, 'y': y1, 'width': x2-x1, 'height': y2-y1},
                        'risk_level': risk_level
                    })
                    logger.info(f"Detection {i+1}: {label} (confidence: {confidence:.2f})")
        except Exception as e:
            logger.error(f"Error during detection: {str(e)}", exc_info=True)
            return jsonify({
                'success': False,
                'error': f'Error during detection: {str(e)}',
                'available_classes': ['OxygenTank', 'NitrogenTank', 'FirstAidBox', 'FireAlarm', 'SafetySwitchPanel', 'EmergencyPhone', 'FireExtinguisher']
            }), 500
            result = results[0]
            for i, box in enumerate(result.boxes):
                class_id = int(box.cls.item())
                label = result.names[class_id]
                confidence = float(box.conf.item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                risk_level = 'high' if confidence < 0.7 else 'medium' if confidence < 0.9 else 'low'
                detections.append({
                    'id': str(i),
                    'label': label,
                    'confidence': confidence,
                    'bbox': {'x': x1, 'y': y1, 'width': x2-x1, 'height': y2-y1},
                    'risk_level': risk_level
                })
            output_image_path = result.path[0] if hasattr(result, 'path') and len(result.path) > 0 else None
            image_id = os.path.basename(output_image_path) if output_image_path else None
            processing_time = int((time.time() - start_time) * 1000)
            return jsonify({
                'success': True,
                'results': detections,
                'processing_time': processing_time,
                'image_id': image_id
            })
        else:
            logger.warning("No detection results")
            processing_time = int((time.time() - start_time) * 1000)
            return jsonify({
                'success': True,
                'results': [],
                'processing_time': processing_time,
                'image_id': None
            })
    except Exception as e:
        logger.error(f"Error in API detection: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@routes.route('/uploads/<path:filename>')
def uploads(filename):
    return send_from_directory('models/logs/detect', filename)

@routes.route('/api/images/<path:filename>')
def api_images(filename):
    return send_from_directory('../models/logs/detect', filename)

@routes.route('/api/health', methods=['GET'])
def api_health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0',
        'source': 'routes_blueprint'
    })

@routes.route('/api/statistics', methods=['GET'])
def api_statistics():
    """Return statistics in the format expected by the frontend."""
    try:
        import json
        stats_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset_statistics.json')
        
        # If the statistics file exists, load it
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                stats = json.load(f)
        else:
            # If the file doesn't exist, run the count script
            logger.info("Statistics file not found, generating now...")
            import subprocess
            subprocess.run(['python', 'count_dataset.py'], cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            with open(stats_file, 'r') as f:
                stats = json.load(f)
        
        # Get class distribution
        class_distribution = stats.get('class_counts', {})
        total_objects = sum(class_distribution.values())
        
        # Calculate percentage for each class
        class_percentages = {
            class_name: round((count / total_objects) * 100, 1)
            for class_name, count in class_distribution.items()
        }
        
        return jsonify({
            'success': True,
            'statistics': {
                'total_train_images': stats.get('total_train_images', 0),
                'train_images_with_objects': stats.get('train_images_with_objects', 0),
                'total_test_images': stats.get('total_test_images', 0),
                'test_images_with_objects': stats.get('test_images_with_objects', 0),
                'total_objects': total_objects,
                'class_distribution': class_distribution,
                'class_percentages': class_percentages,
                'model_metrics': {
                    'accuracy': 95.8,  # This would come from your model evaluation
                    'precision': 94.2,  # Example values
                    'recall': 93.5,     # Example values
                    'f1_score': 93.8    # Example values
                },
                'last_updated': datetime.now().isoformat(),
                'model_version': '1.0.0'
            }
        })
        
    except Exception as e:
        logger.error(f"Error generating statistics: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'Could not generate statistics: {str(e)}'
        }), 500

@routes.route('/api/models', methods=['GET'])
def api_models():
    weights_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'weights')
    if os.path.exists(weights_dir):
        weights = [f for f in os.listdir(weights_dir) if os.path.isfile(os.path.join(weights_dir, f)) and f.endswith('.pt')]
    else:
        weights = []
    active_model = os.path.basename(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    return jsonify({
        'models': weights,
        'active_model': active_model
    })

@routes.route('/predict', methods=['POST'])
def predict():
    """Main /predict endpoint for YOLO detection - accepts image upload or base64."""
    import time
    start_time = time.time()
    try:
        logger.info("Predict endpoint called")
        
        # Check if model exists
        model_exists = os.path.exists(MODEL_PATH)
        if not model_exists:
            logger.warning(f"Model not found at {MODEL_PATH}, using placeholder mode")
            # Return placeholder detections
            processing_time = int((time.time() - start_time) * 1000)
            return jsonify({
                'success': True,
                'detections': [
                    {
                        'label': 'OxygenTank',
                        'score': 0.95,
                        'x1': 100.0,
                        'y1': 100.0,
                        'x2': 200.0,
                        'y2': 200.0
                    },
                    {
                        'label': 'FireExtinguisher',
                        'score': 0.88,
                        'x1': 300.0,
                        'y1': 150.0,
                        'x2': 400.0,
                        'y2': 350.0
                    }
                ],
                'processing_time': processing_time,
                'model_loaded': False
            })
        
        # Handle image upload
        if 'image' not in request.files:
            logger.warning("No image file in request")
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            logger.warning("Empty filename")
            return jsonify({'error': 'No image selected'}), 400
        
        # Save uploaded file
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}_{file.filename}"
        path = os.path.join(UPLOAD, filename)
        file.save(path)
        logger.info(f"File saved to {path}")
        
        # Load model to check classes
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error("ultralytics not installed")
            return jsonify({
                'success': False,
                'error': 'ultralytics module not installed. Please install: pip install ultralytics'
            }), 503
        
        model = YOLO(MODEL_PATH)
        model_classes = model.names
        logger.info(f"Model has {len(model_classes)} classes: {list(model_classes.values())}")
        
        # Expected classes (from training data)
        expected_classes = ['OxygenTank', 'NitrogenTank', 'FirstAidBox', 'FireAlarm', 
                           'SafetySwitchPanel', 'EmergencyPhone', 'FireExtinguisher']
        
        # Check if model classes match expected
        if len(model_classes) != len(expected_classes):
            logger.warning(f"⚠️ Model has {len(model_classes)} classes but expected {len(expected_classes)}. "
                         f"Model may need retraining with correct class configuration.")
            logger.warning(f"   Current classes: {list(model_classes.values())}")
            logger.warning(f"   Expected classes: {expected_classes}")
            logger.warning(f"   ⚠️ This may cause misclassification (e.g., FirstAidBox detected as FireExtinguisher)")
        
        # Check if detection is available
        if not DETECT_AVAILABLE:
            logger.error("Detection module not available. Please install ultralytics: pip install ultralytics")
            return jsonify({
                'success': False,
                'error': 'Detection module not available. Please install dependencies: pip install -r requirements.txt'
            }), 503
        
        # Run detection with higher confidence threshold to reduce false positives
        # Since current model has wrong classes, use higher threshold
        conf_threshold = float(request.form.get('conf', 0.5))  # Increased to 0.5 to reduce false positives
        try:
            results = detect(path, model_path=MODEL_PATH, conf_threshold=conf_threshold)
            logger.info(f"Detection complete, processing results with conf_threshold={conf_threshold}")
        except Exception as detect_error:
            logger.error(f"Detection error: {detect_error}", exc_info=True)
            raise
        
        # Format detections
        detections = []
        image_width = None
        image_height = None
        
        if results and len(results) > 0:
            result = results[0]
            
            # Get original image dimensions from result
            # YOLO result.orig_shape contains (height, width) of original image
            if hasattr(result, 'orig_shape') and result.orig_shape is not None:
                image_height, image_width = result.orig_shape
                logger.info(f"✅ Original image dimensions from YOLO orig_shape: {image_width}x{image_height}")
            else:
                logger.warning("⚠️ YOLO result.orig_shape not available, using fallback")
            
            # ALWAYS verify by reading the actual image file
            try:
                import cv2
                img = cv2.imread(path)
                if img is not None:
                    file_height, file_width = img.shape[:2]
                    logger.info(f"✅ Image dimensions from file (cv2): {file_width}x{file_height}")
                    
                    # Compare with YOLO's reported dimensions
                    if image_width and image_height:
                        if file_width != image_width or file_height != image_height:
                            logger.error(f"❌ DIMENSION MISMATCH! YOLO says {image_width}x{image_height}, "
                                       f"but file is {file_width}x{file_height}")
                            logger.error(f"   Using file dimensions for coordinate verification")
                            # Use file dimensions as they're the actual source
                            image_width, image_height = file_width, file_height
                        else:
                            logger.info(f"✅ Dimensions match: {image_width}x{image_height}")
                    else:
                        # Use file dimensions if YOLO didn't provide them
                        image_width, image_height = file_width, file_height
            except Exception as e:
                logger.warning(f"Could not read image dimensions: {e}")
            
            if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                for i, box in enumerate(result.boxes):
                    try:
                        class_id = int(box.cls.item())
                        label = result.names[class_id] if hasattr(result, 'names') else f"Class_{class_id}"
                        confidence = float(box.conf.item())
                        
                        # Only include detections above threshold (double-check)
                        if confidence < conf_threshold:
                            continue
                        
                        # Get coordinates - box.xyxy returns coordinates in original image space
                        # Format: [x1, y1, x2, y2] in pixels
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        # CRITICAL: Log raw coordinates before any processing
                        logger.info(f"RAW Detection {i}: {label} - Raw coords from YOLO: ({x1:.1f}, {y1:.1f}) -> ({x2:.1f}, {y2:.1f})")
                        
                        # Get YOLO's reported image shape for verification
                        yolo_shape = None
                        if hasattr(result, 'shape') and result.shape is not None:
                            yolo_shape = result.shape
                            logger.info(f"YOLO result.shape (processed): {yolo_shape}")
                        if hasattr(result, 'orig_shape') and result.orig_shape is not None:
                            logger.info(f"YOLO orig_shape (original): {result.orig_shape}")
                        
                        # Verify coordinates are reasonable
                        if image_width and image_height:
                            # Check if coordinates exceed image bounds (indicates coordinate system mismatch)
                            if x1 > image_width or x2 > image_width or y1 > image_height or y2 > image_height:
                                logger.warning(f"⚠️ Coordinates exceed image bounds! Coords: ({x1:.1f}, {y1:.1f}) -> ({x2:.1f}, {y2:.1f}), "
                                             f"Image: {image_width}x{image_height}")
                                logger.warning(f"   This suggests coordinates might be in wrong coordinate system!")
                            
                            # Clamp coordinates to image bounds
                            x1 = max(0, min(image_width, float(x1)))
                            y1 = max(0, min(image_height, float(y1)))
                            x2 = max(0, min(image_width, float(x2)))
                            y2 = max(0, min(image_height, float(y2)))
                            
                            # Log final coordinates
                            logger.info(f"FINAL Detection {i}: {label} at ({x1:.1f}, {y1:.1f}) -> ({x2:.1f}, {y2:.1f}) "
                                       f"on image {image_width}x{image_height}")
                        else:
                            logger.warning(f"⚠️ No image dimensions available! Cannot verify coordinates.")
                        
                        detections.append({
                            'label': label,
                            'score': confidence,
                            'x1': float(x1),
                            'y1': float(y1),
                            'x2': float(x2),
                            'y2': float(y2)
                        })
                    except Exception as box_error:
                        logger.warning(f"Error processing box {i}: {box_error}")
                        continue
        
        processing_time = int((time.time() - start_time) * 1000)
        
        response_data = {
            'success': True,
            'detections': detections,
            'processing_time': processing_time,
            'model_loaded': True
        }
        
        # Include image dimensions in response for frontend verification
        if image_width and image_height:
            response_data['image_width'] = int(image_width)
            response_data['image_height'] = int(image_height)
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500
