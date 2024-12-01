import numpy as np
from scipy.signal import convolve2d
from skimage import io, color, img_as_ubyte
from skimage.restoration import unsupervised_wiener
from numpy.fft import fft2, ifft2
from io import BytesIO
from PIL import Image

def wiener_from_scratch(file, K=10):
    img = io.imread(file)
    if len(img.shape) == 3:
        img = color.rgb2gray(img)
    kernel = np.ones((5, 5)) / 25
    dummy_fft = fft2(img)
    kernel_fft = fft2(np.pad(kernel, [(0, img.shape[0] - kernel.shape[0]),
                                      (0, img.shape[1] - kernel.shape[1])], 'constant'))
    kernel_power = np.abs(kernel_fft) ** 2
    H = np.conj(kernel_fft) / (kernel_power + K)
    restored = np.abs(ifft2(dummy_fft * H))
    restored_normalized = np.uint8((restored / np.max(restored)) * 255)

    buf = BytesIO()
    Image.fromarray(restored_normalized).save(buf, format='PNG')
    buf.seek(0)
    return buf

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
