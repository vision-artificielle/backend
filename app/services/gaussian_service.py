import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
from skimage import io, color, img_as_ubyte
from io import BytesIO
from PIL import Image

import tempfile
import cv2

def gaussian_kernel(size, sigma=1):
    """
    Crée un noyau gaussien 2D.
    """
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)

def gaussian_filter_from_scratch(file, size=5, sigma=1):
    """
    Applique un filtre gaussien sur une image donnée (from scratch).
    """
    img = io.imread(file)
    if len(img.shape) == 3:
        img = color.rgb2gray(img)
    kernel = gaussian_kernel(size, sigma)
    filtered_image = convolve2d(img, kernel, mode='same', boundary='wrap')
    filtered_image = img_as_ubyte(filtered_image / np.max(filtered_image))  # Normalisation

    buf = BytesIO()
    Image.fromarray(filtered_image).save(buf, format='PNG')
    buf.seek(0)
    return buf

def gaussian_filter_predefined(file, sigma=1):
    """
    Applique un filtre gaussien prédéfini sur une image donnée.
    """
    img = io.imread(file)
    if len(img.shape) == 3:
        img = color.rgb2gray(img)
    filtered_image = gaussian_filter(img, sigma=sigma)
    filtered_image = img_as_ubyte(filtered_image)

    buf = BytesIO()
    Image.fromarray(filtered_image).save(buf, format='PNG')
    buf.seek(0)
    return buf



def apply_gaussian_filter_predefined(image_stream, psf_kernel_size=5, noise_std=0.05, sigma=1):
    """
    Applique un filtre Gaussien prédéfini à une image envoyée.
    """
    # Charger l'image depuis le flux
    image = io.imread(image_stream)

    # Convertir en niveaux de gris si l'image est en couleur
    if len(image.shape) == 3:
        image = color.rgb2gray(image)

    # Normalisation de l'image entre 0 et 1
    image = image / 255.0

    # Ajouter du flou et du bruit
    psf = np.ones((psf_kernel_size, psf_kernel_size)) / (psf_kernel_size ** 2)  # Noyau de flou (PSF)
    blurred = convolve2d(image, psf, mode='same', boundary='wrap')
    noise = np.random.normal(0, noise_std, blurred.shape)  # Bruit gaussien
    blurred_noisy = blurred + noise

    # Appliquer le filtre Gaussien
    filtered_image = gaussian_filter(blurred_noisy, sigma=sigma)

    # Convertir l'image filtrée en format 8-bit
    filtered_image = img_as_ubyte(filtered_image / np.max(filtered_image))

    # Sauvegarde de l'image filtrée dans un fichier temporaire
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    cv2.imwrite(temp_file.name, filtered_image)

    return temp_file.name