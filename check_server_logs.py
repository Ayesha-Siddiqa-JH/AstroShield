#!/usr/bin/env python3
"""Check what error the server is producing."""
import requests
import json

# Test with a simple request to see the actual error
try:
    # Create a minimal test image
    import io
    from PIL import Image
    
    # Create a simple test image
    img = Image.new('RGB', (100, 100), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    files = {'image': ('test.png', img_bytes, 'image/png')}
    response = requests.post('http://127.0.0.1:10000/predict', files=files, timeout=30)
    
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n✅ Success!")
        print(json.dumps(data, indent=2))
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

