#!/usr/bin/env python3
"""Start the backend server."""
from app.backend import app
import os

port = int(os.environ.get('PORT', 10000))
print(f"Starting server on port {port}...")
app.run(host='0.0.0.0', port=port, debug=True, threaded=True, use_reloader=False)

