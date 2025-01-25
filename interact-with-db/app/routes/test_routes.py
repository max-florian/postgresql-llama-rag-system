from flask import Blueprint, jsonify
from app.controllers.chatbot_controller import test_ollama

test_bp = Blueprint("test", __name__)

@test_bp.route("/test-ollama", methods=["GET"])
def test_ollama_route():
    try:
        return test_ollama()
    except Exception as e:
        return jsonify({"error": str(e)}), 500
