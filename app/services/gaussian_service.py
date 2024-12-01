import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
from skimage import io, color, img_as_ubyte
from io import BytesIO
from PIL import Image

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
