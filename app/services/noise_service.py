import numpy as np
from scipy.signal import convolve2d
from skimage import io, color, img_as_ubyte
from io import BytesIO
from PIL import Image

def add_noise(file, noise_level):
    # Read and process the image
    img = io.imread(file)

    # Convert to grayscale if it's a color image
    if len(img.shape) == 3:
        img = color.rgb2gray(img)

    # Apply a simple blur (PSF: Point Spread Function)
    psf = np.ones((5, 5)) / 25
    blurred = convolve2d(img, psf, mode='same', boundary='wrap')

    # Add Gaussian noise
    noise = np.random.normal(0, noise_level, blurred.shape)
    noisy_image = img_as_ubyte((blurred + noise) / np.max(blurred + noise))

    # Save the noisy image to a buffer
    buf = BytesIO()
    Image.fromarray(noisy_image).save(buf, format='PNG')
    buf.seek(0)
    return buf
