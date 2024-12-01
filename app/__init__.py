from flask import Flask

def create_app():
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = './uploads'
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limite de 16 MB

    with app.app_context():
        from .routes import app_routes
        app.register_blueprint(app_routes)

    return app