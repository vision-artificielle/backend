import numpy as np
from PIL import Image

def load_and_preprocess_image(image_stream):
    """Charge et prétraite l'image depuis un flux binaire."""
    img = Image.open(image_stream).convert("RGB")
    img = img.resize((224, 224))  # Redimensionner à 224x224 comme dans le notebook
    img_array = np.array(img) / 255.0  # Normalisation entre 0 et 1

    print(f"Image chargée - dtype: {img_array.dtype}, shape: {img_array.shape}")
    return img_array
