# from flask import Flask, request, jsonify, send_file
# import numpy as np
# import cv2
# from io import BytesIO
# from PIL import Image
# import os
#
# app = Flask(__name__)
#
# # Dossier temporaire pour stocker les images traitées
# UPLOAD_FOLDER = "./uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
#
#
# def load_and_preprocess_image(image_stream):
#     """Charge et prétraite l'image depuis un flux binaire."""
#     img = Image.open(image_stream).convert("RGB")
#     img = img.resize((224, 224))  # Redimensionner à 224x224 comme dans le notebook
#     img_array = np.array(img) / 255.0  # Normalisation entre 0 et 1
#
#     print(f"Image chargée - dtype: {img_array.dtype}, shape: {img_array.shape}")
#     return img_array
#
#
#
# def denoise_image(image):
#     """La fonction de débruitage."""
#     # Vérifier que l'image est en RGB
#     if len(image.shape) == 2:  # Image en niveaux de gris
#         image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
#
#     # Vérifier que l'image a 3 ou 4 canaux
#     if image.shape[-1] not in [3, 4]:
#         raise ValueError("L'image doit avoir 3 (RGB) ou 4 (RGBA) canaux")
#
#     # Convertir en uint8
#     image_uint8 = (image * 255).astype(np.uint8)
#
#     # Appliquer le débruitage
#     denoised_image = cv2.fastNlMeansDenoisingColored(image_uint8, None, 10, 10, 7, 21)
#
#     return denoised_image
#
#
#
#
# @app.route("/denoise-opencv", methods=["POST"])
# def denoise():
#     """API pour débruiter une image."""
#     if "file" not in request.files:
#         return jsonify({"error": "No file part"}), 400
#
#     file = request.files["file"]
#     if file.filename == "":
#         return jsonify({"error": "No selected file"}), 400
#
#     try:
#         # Charger et prétraiter l'image
#         image_array = load_and_preprocess_image(file.stream)
#
#         # Débruitage
#         denoised_image = denoise_image(image_array)
#
#         # Sauvegarde temporaire pour envoi
#         output_path = os.path.join(UPLOAD_FOLDER, "denoised_image.png")
#         cv2.imwrite(output_path, cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB))  # BGR vers RGB
#
#         return send_file(output_path, mimetype="image/png")
#
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#
#
# if __name__ == "__main__":
#     app.run(debug=True)
