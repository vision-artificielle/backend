import cv2
from flask import Blueprint, request, send_file, jsonify
from io import BytesIO
from PIL import Image

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
    file = request.files['image']
    sigma = float(request.form.get('sigma', 1))
    noise_level = float(request.form.get('noise_level', 0.1))  # Noise level

    # Add noise
    noisy_image = add_noise(file, noise_level)

    # Apply Gaussian filter
    filtered_image = gaussian_filter_predefined(noisy_image, sigma=sigma)
    return send_file(filtered_image, mimetype='image/png')


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
    file = request.files['image']
    noise_level = float(request.form.get('noise_level', 0.1))  # Noise level

    # Add noise
    noisy_image = add_noise(file, noise_level)

    # Apply Wiener filter from scratch
    restored_image = wiener_from_scratch(noisy_image)
    return send_file(restored_image, mimetype='image/png')


@app_routes.route('/wiener_predefined', methods=['POST'])
def route_wiener_predefined():
    file = request.files['image']
    noise_level = float(request.form.get('noise_level', 0.1))  # Noise level

    # Add noise
    noisy_image = add_noise(file, noise_level)

    # Apply Wiener filter (predefined)
    restored_image = wiener_predefined(noisy_image)
    return send_file(restored_image, mimetype='image/png')
