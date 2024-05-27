from flask import Flask, request, jsonify
from deepface import DeepFace
import cv2
import numpy as np
import requests

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze_image():
    try:
        # Check if request is in JSON format.
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.get_json()

        # Check that JSON file contains necessary keys, that is image.
        if 'image' not in data:
            return jsonify({"error": "Invalid JSON format"}), 400
        
        image_url = data['image']
        
        # Download the image from the URL
        response = requests.get(image_url)
        if response.status_code != 200:
            return jsonify({"error": "Failed to download image"}), 400
        
        image_bytes = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_UNCHANGED)
        
        #If there's no decodable image, raise error.
        if image is None:
            return jsonify({"error": "Failed to decode the image"}), 400
        
        # Analyze the image.
        predictions = DeepFace.analyze(image, actions=['emotion'])
        
        # Check the results returned from DeepFace and return the results
        if len(predictions) == 0:
            return jsonify({"error": "Yüz tespit edilemedi!"}), 400
        elif len(predictions) > 1:
            return jsonify({"error": "Birden fazla yüz bulundu!"}), 400
        else:
            return jsonify(predictions[0]['dominant_emotion']), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)