import numpy as np
from scipy.signal import convolve2d
from skimage import io, color, img_as_ubyte
from skimage.restoration import unsupervised_wiener
from numpy.fft import fft2, ifft2
from io import BytesIO
from PIL import Image

def wiener_from_scratch(img, kernel, K=10):
    """
    Applique un filtre de Wiener à une image donnée.

    :param img: Image bruitée
    :param kernel: Noyau de flou (PSF)
    :param K: Paramètre de régularisation
    :return: Image restaurée
    """
    # Copier l'image pour travailler sur une copie
    dummy = np.copy(img)

    # Ajouter du padding au noyau pour le faire correspondre à la taille de l'image
    kernel_padded = np.pad(kernel, [(0, dummy.shape[0] - kernel.shape[0]),
                                    (0, dummy.shape[1] - kernel.shape[1])], 'constant')

    # Transformation de Fourier de l'image et du noyau
    dummy_fft = fft2(dummy)
    kernel_fft = fft2(kernel_padded)

    # Calcul de la puissance du noyau
    kernel_power = np.abs(kernel_fft) ** 2

    # Calcul du filtre de Wiener
    H = np.conj(kernel_fft) / (kernel_power + K)

    # Appliquer le filtre
    restored_fft = dummy_fft * H

    # Retour à l'espace des images avec l'inverse de la transformée de Fourier
    restored = np.abs(ifft2(restored_fft))

    # Normaliser entre 0 et 255
    restored_normalized = restored / np.max(restored)
    restored_normalized = np.uint8(restored_normalized * 255)

    return restored_normalized

def wiener_predefined(file):
    img = io.imread(file)
    if len(img.shape) == 3:
        img = color.rgb2gray(img)
    kernel = np.ones((5, 5)) / 25
    restored, _ = unsupervised_wiener(img, kernel)
    restored_image = img_as_ubyte(restored)

    buf = BytesIO()
    Image.fromarray(restored_image).save(buf, format='PNG')
    buf.seek(0)
    return buf
