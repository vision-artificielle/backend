from flask import Blueprint, request, send_file
from app.services.noise_service import add_noise
from app.services.wiener_service import wiener_from_scratch, wiener_predefined

app_routes = Blueprint('app_routes', __name__)

@app_routes.route('/add_noise', methods=['POST'])
def route_add_noise():
    file = request.files['image']
    noise_level = float(request.form.get('noise_level', 0.1))
    noisy_image = add_noise(file, noise_level)
    return send_file(noisy_image, mimetype='image/png')

@app_routes.route('/wiener_from_scratch', methods=['POST'])
def route_wiener_from_scratch():
    file = request.files['image']
    restored_image = wiener_from_scratch(file)
    return send_file(restored_image, mimetype='image/png')

@app_routes.route('/wiener_predefined', methods=['POST'])
def route_wiener_predefined():
    file = request.files['image']
    restored_image = wiener_predefined(file)
    return send_file(restored_image, mimetype='image/png')
