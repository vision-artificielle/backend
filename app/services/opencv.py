from io import BytesIO
from PIL import Image
import cv2
import numpy as np

from app.utils import load_and_preprocess_image


def denoise_image(image):
    """La fonction de débruitage."""
    # Vérifier que l'image est en RGB
    if len(image.shape) == 2:  # Image en niveaux de gris
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Vérifier que l'image a 3 ou 4 canaux
    if image.shape[-1] not in [3, 4]:
        raise ValueError("L'image doit avoir 3 (RGB) ou 4 (RGBA) canaux")

    # Convertir en uint8
    image_uint8 = (image * 255).astype(np.uint8)

    # Appliquer le débruitage
    denoised_image = cv2.fastNlMeansDenoisingColored(image_uint8, None, 10, 10, 7, 21)

    return denoised_image

def process_and_denoise_image(image_stream):
    """
    Charge, débruite, et convertit l'image en fichier PNG en mémoire.
    """
    # Charger et prétraiter l'image
    image_array = load_and_preprocess_image(image_stream)

    # Appliquer le débruitage
    denoised_image = denoise_image(image_array)

    # Convertir l'image débruitée en un fichier PNG en mémoire
    image_io = BytesIO()
    denoised_pil_image = Image.fromarray(cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB))
    denoised_pil_image.save(image_io, format="PNG")
    image_io.seek(0)

    return image_io