from flask import Flask
from app.routes.query_routes import query_bp
from app.routes.test_routes import test_bp
from app.utils.logging_config import configure_logging

def create_app():
    app = Flask(__name__)
    configure_logging()
    
    # Register blueprints
    app.register_blueprint(query_bp)
    app.register_blueprint(test_bp)
    
    return app
