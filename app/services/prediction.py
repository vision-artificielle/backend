import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import base64

# Charger le modèle une seule fois
model = tf.keras.models.load_model('/Users/aya/Desktop/academic/vision-artificielle/projet1/backend/app/model/denoising_model.h5')

def predict_model(image_file):
    """
    Process the image: load, resize, predict, and return binary data.
    """
    try:
        img = Image.open(image_file).convert('L')  # Convert to grayscale
        img = img.resize((128, 128))  # Resize to match training dimensions
        img_array = np.array(img).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=(0, -1))  # Add necessary dimensions

        # Predict with the model
        denoised_img = model.predict(img_array)
        denoised_img = denoised_img[0].squeeze() * 255.0  # Rescale to [0, 255]

        # Convert to uint8 image
        denoised_img = denoised_img.astype(np.uint8)
        _, buffer = cv2.imencode('.png', denoised_img)
        return buffer.tobytes()  # Return binary data
    except Exception as e:
        raise ValueError(f"Image processing error: {str(e)}")

