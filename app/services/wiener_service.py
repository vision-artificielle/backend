import numpy as np
from scipy.signal import convolve2d
from skimage import io, color, img_as_ubyte
from skimage.restoration import unsupervised_wiener
from numpy.fft import fft2, ifft2
from io import BytesIO
from PIL import Image

def wiener_from_scratch(image_array, K=10):
    """Apply Wiener filter from scratch to a NumPy array."""
    # Convert to grayscale if the image is in color
    if len(image_array.shape) == 3:
        image_array = color.rgb2gray(image_array)

    # Normalize the image
    image_array = image_array / 255.0

    # Create a kernel (e.g., averaging kernel)
    kernel = np.ones((5, 5)) / 25

    # Perform FFT
    image_fft = fft2(image_array)
    kernel_padded = np.pad(kernel, [(0, image_array.shape[0] - kernel.shape[0]),
                                    (0, image_array.shape[1] - kernel.shape[1])], 'constant')
    kernel_fft = fft2(kernel_padded)

    # Wiener filter
    kernel_power = np.abs(kernel_fft) ** 2
    H = np.conj(kernel_fft) / (kernel_power + K)
    restored = np.abs(ifft2(image_fft * H))

    # Normalize the restored image
    restored_normalized = np.uint8((restored / np.max(restored)) * 255)

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
