from flask import Blueprint, request, send_file, jsonify, make_response

from app.services.gaussian_service import gaussian_filter_from_scratch, apply_gaussian_filter_predefined
from app.services.noise_service import add_noise
from app.services.opencv import process_and_denoise_image
from app.services.prediction import predict_model
from app.services.wiener_service import  apply_wiener_filter, apply_wiener_filter_predefined

app_routes = Blueprint('app_routes', __name__)


@app_routes.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to denoise an image.
    """
    try:
        file = request.files['image']
        if not file:
            return jsonify({"error": "No image provided"}), 400

        # Generate the denoised image
        denoised_image = predict_model(file)

        # Send image as binary response
        response = make_response(denoised_image)
        response.headers['Content-Type'] = 'image/png'
        return response
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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
    """
    Endpoint pour appliquer le filtre Gaussien prédéfini à une image envoyée.
    """
    if 'image' not in request.files:
        return {"error": "No image file provided"}, 400

    file = request.files['image']
    try:
        # Utiliser la fonction pour appliquer le filtre Gaussien
        filtered_image_path = apply_gaussian_filter_predefined(file.stream)

        # Retourner l'image traitée au client
        return send_file(filtered_image_path, mimetype='image/png', as_attachment=True, download_name="filtered_image.png")

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app_routes.route("/opencv", methods=["POST"])
def denoise():
    """API pour débruiter une image."""
    if "image" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        # Utilisation de la nouvelle fonction pour charger, débruiter et convertir l'image
        image_io = process_and_denoise_image(file.stream)

        # Retourner l'image débruitée
        return send_file(image_io, mimetype="image/png")
    except Exception as e:
        return jsonify({"error": str(e)}), 500




@app_routes.route('/wiener_from_scratch', methods=['POST'])
def route_wiener_from_scratch():
    """
    Endpoint pour appliquer le filtre de Wiener à une image envoyée.
    """
    if 'image' not in request.files:
        return "No image file provided", 400

    file = request.files['image']
    if not file:
        return "No file uploaded", 400

    try:
        # Utilisation de la fonction séparée pour appliquer le filtre de Wiener
        output = apply_wiener_filter(file.stream)

        # Retourner l'image traitée
        return send_file(output, mimetype='image/png', as_attachment=False, download_name='restored_image.png')

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app_routes.route('/wiener_predefined', methods=['POST'])
def route_wiener_predefined():
    """
    Endpoint pour appliquer le filtre de Wiener prédéfini à une image envoyée.
    """
    if 'image' not in request.files:
        return "No image file provided", 400

    file = request.files['image']
    if not file:
        return "No file uploaded", 400

    try:
        # Utilisation de la fonction séparée pour appliquer le filtre de Wiener
        output = apply_wiener_filter_predefined(file.stream)

        # Retourner l'image traitée
        return send_file(output, mimetype='image/png', as_attachment=False, download_name='restored_image.png')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

