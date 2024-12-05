import cv2
from flask import Blueprint, request, send_file, jsonify
from io import BytesIO
import zipfile
from PIL import Image
from flask import Flask, request, send_file
import numpy as np
from skimage import io, color, img_as_ubyte
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
import cv2
import tempfile
import os

from app.services.gaussian_service import gaussian_filter_from_scratch, gaussian_filter_predefined
from app.services.noise_service import add_noise
from app.services.opencv import denoise_image
from app.services.wiener_service import wiener_from_scratch, wiener_predefined
from app.utils import load_and_preprocess_image

app_routes = Blueprint('app_routes', __name__)


@app_routes.route('/add_noise', methods=['POST'])
def route_add_noise():
    try:
        # Retrieve image and noise level from request
        file = request.files.get('image')
        noise_level = float(request.form.get('noise_level', 0.1))

        if not file:
            return jsonify({"error": "No image file provided"}), 400

        if noise_level < 0 or noise_level > 1:
            return jsonify({"error": "Noise level must be between 0 and 1"}), 400

        # Process the image with noise
        noisy_image = add_noise(file, noise_level)
        return send_file(noisy_image, mimetype='image/png')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app_routes.route('/gaussian_from_scratch', methods=['POST'])
def route_gaussian_from_scratch():
    file = request.files['image']
    size = int(request.form.get('size', 5))
    sigma = float(request.form.get('sigma', 1))
    noise_level = float(request.form.get('noise_level', 0.1))  # Noise level

    # Add noise
    noisy_image = add_noise(file, noise_level)

    # Apply Gaussian filter
    filtered_image = gaussian_filter_from_scratch(noisy_image, size=size, sigma=sigma)
    return send_file(filtered_image, mimetype='image/png')


@app_routes.route('/gaussian_predefined', methods=['POST'])
def route_gaussian_predefined():
    # file = request.files['image']
    # sigma = float(request.form.get('sigma', 1))
    # noise_level = float(request.form.get('noise_level', 0.2))  # Noise level
    #
    # # Add noise
    # noisy_image = add_noise(file, noise_level)
    #
    # # Apply Gaussian filter
    # filtered_image = gaussian_filter_predefined(noisy_image, sigma=sigma)
    # return send_file(filtered_image, mimetype='image/png')
    # Vérifie si une image est envoyée
    # Vérifie si une image est envoyée
    if 'image' not in request.files:
        return {"error": "No image file provided"}, 400

    # Récupère l'image envoyée
    file = request.files['image']
    image = io.imread(file)

    # Convertit en niveaux de gris si l'image est en couleur
    if len(image.shape) == 3:
        image = color.rgb2gray(image)

    # Normalise l'image entre 0 et 1 (pour éviter les erreurs de calcul)
    image = image / 255.0

    # Ajouter du flou et du bruit
    psf = np.ones((5, 5)) / 25  # Noyau de flou (PSF)
    blurred = convolve2d(image, psf, mode='same', boundary='wrap')
    noise = np.random.normal(0, 0.05, blurred.shape)  # Bruit gaussien (réduit)
    blurred_noisy = blurred + noise

    # Applique le filtre prédéfini (Filtre Gaussien)
    filtered_image = gaussian_filter(blurred_noisy, sigma=1)

    # Remet l'image dans un format 8-bit pour l'affichage
    filtered_image = img_as_ubyte(filtered_image / np.max(filtered_image))

    # Sauvegarde l'image filtrée dans un fichier temporaire
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    cv2.imwrite(temp_file.name, filtered_image)

    # Retourne l'image traitée au client
    return send_file(temp_file.name, mimetype='image/png', as_attachment=True, download_name="filtered_image.png")


@app_routes.route("/opencv", methods=["POST"])
def denoise():
    """API pour débruiter une image."""
    if "image" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        # Charger et prétraiter l'image
        image_array = load_and_preprocess_image(file.stream)

        # Débruitage
        denoised_image = denoise_image(image_array)

        # Convertir l'image débruitée en fichier
        image_io = BytesIO()
        denoised_pil_image = Image.fromarray(cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB))
        denoised_pil_image.save(image_io, format="PNG")
        image_io.seek(0)

        return send_file(image_io, mimetype="image/png")
    except Exception as e:
        return jsonify({"error": str(e)}), 500




@app_routes.route('/wiener_from_scratch', methods=['POST'])
def route_wiener_from_scratch():
    """Apply Wiener filter and return both noisy and restored images."""
    try:
        file = request.files['image']
        noise_level = float(request.form.get('noise_level', 0.2))  # Noise level

        # Add noise
        noisy_image = add_noise(file, noise_level)

        # Apply Wiener filter
        restored_image = wiener_from_scratch(noisy_image)

        # Convert images to PIL format and save to BytesIO
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zf:
            # Save noisy image
            noisy_image_io = BytesIO()
            noisy_pil_image = Image.fromarray(noisy_image)
            noisy_pil_image.save(noisy_image_io, format='PNG')
            noisy_image_io.seek(0)
            zf.writestr('noisy_image.png', noisy_image_io.read())

            # Save restored image
            restored_image_io = BytesIO()
            restored_pil_image = Image.fromarray(restored_image)
            restored_pil_image.save(restored_image_io, format='PNG')
            restored_image_io.seek(0)
            zf.writestr('restored_image.png', restored_image_io.read())

        # Reset buffer position
        zip_buffer.seek(0)

        # Send ZIP file as response
        return send_file(zip_buffer, mimetype='application/zip', as_attachment=True, download_name='images.zip')

    except Exception as e:
        return jsonify({"error": str(e)}), 500




@app_routes.route('/wiener_predefined', methods=['POST'])
def route_wiener_predefined():
    file = request.files['image']
    noise_level = float(request.form.get('noise_level', 0.1))  # Noise level

    # Add noise
    noisy_image = add_noise(file, noise_level)

    # Apply Wiener filter (predefined)
    restored_image = wiener_predefined(noisy_image)
    return send_file(restored_image, mimetype='image/png')
