import cv2
import numpy as np

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