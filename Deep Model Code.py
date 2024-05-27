from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('happyandsad.jpg')

try:
    # Analyze the image using DeepFace
    predictions = DeepFace.analyze(img)

    # Check the number of detected faces and handle accordingly
    if len(predictions) > 1:
        raise Exception("Birden fazla görüntü!")  # More than one face detected
    elif len(predictions) == 1:
        print(predictions)  # Print the analysis of the detected face
except Exception as e:
    # Handle specific exceptions related to face detection
    if "Face could not be detected" in str(e):
        print("Resimde yüz bulunamadı!")  # No face found in the image
    else:
        print("Hata: ", e)  # Print any other errors
