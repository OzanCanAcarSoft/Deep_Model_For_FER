import unittest
from unittest.mock import patch
from flask import Flask, request, jsonify
from deepface import DeepFace
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze_image():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file found in the request"}), 400

        # Acquire the image sent with the request
        image_file = request.files['image']
        image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None:
            return jsonify({"error": "Failed to decode the image"}), 400
        
        # Analyze the Image
        predictions = DeepFace.analyze(image)
        
        # Return proper result
        if len(predictions) == 0:
            return jsonify({"error": "No Face Detected!"}), 400
        elif len(predictions) > 1:
            return jsonify({"error": "More than one faces detected!"}), 400
        else:
            return jsonify(predictions), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

class TestAnalyzeImage(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
    
    #Test each case

    @patch('deepface.DeepFace.analyze')
    def test_no_image(self, mock_analyze):
        response = self.app.post('/analyze')
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, {'error': 'No image file found in the request'})
        mock_analyze.assert_not_called()

    @patch('deepface.DeepFace.analyze')
    def test_invalid_image(self, mock_analyze):
        with open('api.py', 'rb') as image_file:
            response = self.app.post('/analyze', data={'image': image_file})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, {'error': 'Failed to decode the image'})
        mock_analyze.assert_not_called()

    @patch('deepface.DeepFace.analyze')
    def test_no_face(self, mock_analyze):
        mock_analyze.return_value = []
        with open('car.jpeg', 'rb') as image_file:
            response = self.app.post('/analyze', data={'image': image_file})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, {'error': 'No Face Detected!'})
        mock_analyze.assert_called_once()

    @patch('deepface.DeepFace.analyze')
    def test_multiple_faces(self, mock_analyze):
        mock_analyze.return_value = [{}] * 2
        with open('happyandsad.jpg', 'rb') as image_file:
            response = self.app.post('/analyze', data={'image': image_file})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, {'error': 'More than one faces detected!'})
        mock_analyze.assert_called_once()

    @patch('deepface.DeepFace.analyze')
    def test_success(self, mock_analyze):
        mock_analyze.return_value = [{'age': 30, 'gender': 'Male'}]
        with open('happyboy.jpeg', 'rb') as image_file:
            response = self.app.post('/analyze', data={'image': image_file})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, [{'age': 30, 'gender': 'Male'}])
        mock_analyze.assert_called_once()

if __name__ == '__main__':
    unittest.main()