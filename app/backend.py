from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import os
import sys
from datetime import datetime

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Create the Flask application
app = Flask(__name__)

# Configure CORS for frontend connection
CORS(app, 
     origins=[

         'http://localhost:3000',
         'http://localhost:8080',
         'http://localhost:8000',
         'http://127.0.0.1:8000',
         'http://127.0.0.1:10000'
     ],
     methods=['GET', 'POST', 'OPTIONS'],
     allow_headers=['Content-Type', 'Authorization'])



# Import and register routes Blueprint
try:
    from app.routes import routes  # Correct import for Render and most production setups
    app.register_blueprint(routes)
    logger.info("✅ Routes Blueprint registered successfully")
    
    # Register training routes
    try:
        from app.routes_training import training_routes
        app.register_blueprint(training_routes)
        logger.info("✅ Training routes registered successfully")
    except ImportError:
        logger.warning("⚠️ Training routes not available (optional)")
except ImportError as e:
    logger.error(f"❌ Could not import routes Blueprint: {e}")
    # Fallback routes...
    # Create fallback routes
    @app.route('/')
    def home():
        """Serve the main HTML page as fallback."""
        try:
            from flask import render_template
            return render_template('index.html')
        except Exception:
            return jsonify({
                'message': 'AstroShield Backend is running',
                'status': 'healthy',
                'version': '1.0.0',
                'note': 'Routes module not loaded. Please install dependencies: pip install -r requirements.txt'
            })
    
    @app.route('/favicon.ico')
    def favicon():
        """Handle favicon requests to prevent 404 errors."""
        return '', 204  # No content, but successful
    
    @app.route('/api/health')
    def health():
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0'
        })
    



    logger.warning("⚠️ Using fallback routes due to import error")

# Error handler
@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {e}", exc_info=True)
    import traceback
    error_details = traceback.format_exc()
    logger.error(f"Traceback: {error_details}")
    return jsonify({
        'error': 'Internal server error',
        'message': str(e) if app.debug else 'An error occurred',
        'traceback': error_details if app.debug else None
    }), 500

# Basic health check endpoint (backup in case routes.py is not available)
@app.route('/health', methods=['GET'])
def health_check():
    logger.info("Health check requested")
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'port': os.environ.get('PORT', 'unknown'),
        'version': '1.0.0'
    }), 200

# CORS preflight handler
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = jsonify({'status': 'ok'})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "*")
        response.headers.add('Access-Control-Allow-Methods', "*")
        return response

# Get port from environment
port = int(os.environ.get('PORT', 10000))
logger.info(f"AstroShield Backend initialized on port {port}")

# Application factory pattern (good practice)
def create_app():
    return app

if __name__ == '__main__':
    logger.info(f"Starting AstroShield backend server on port {port}")
    app.config['DEBUG'] = True
    app.run(host='0.0.0.0', port=port, debug=True, threaded=True)