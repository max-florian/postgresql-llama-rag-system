import requests
from flask import jsonify
import json
from app.controllers.embedding_controller import get_embedding
from app.database.connection import get_cursor

OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "llama3.2:1b"

def query_processing(user_query):
    relevant_docs = query_relevant_docs(user_query)
    if not relevant_docs:
        return "I'm sorry, I couldn't find any relevant information."

    doc_text = relevant_docs[0][0]
    payload = {
        "model": MODEL_NAME,
        "prompt": f"This is the relevant information: {doc_text}. This is the user question: {user_query}. Generate a precise response"
    }
    response = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, stream=True)

    response_text = ""
    for line in response.iter_lines():
        if line:
            json_line = json.loads(line)
            if "response" in json_line:
                response_text += json_line["response"]
    return response_text.strip()

def query_relevant_docs(user_query, top_k=1):
    query_embedding = get_embedding(user_query)
    sql = """
        SELECT doc_text, (embedding <=> %s::vector) AS distance
        FROM doc
        ORDER BY distance
        LIMIT %s;
    """
    cursor = get_cursor()
    cursor.execute(sql, (query_embedding, top_k))
    return cursor.fetchall()

def test_ollama():
    payload = {
        "model": MODEL_NAME,
        "prompt": "Hello, Ollama!"
    }
    response = requests.post(f"{OLLAMA_URL}/api/generate", json=payload)
    if response.status_code == 200:
        return jsonify({"status": "success", "message": "Ollama communication successful."})
    else:
        raise Exception(response.text)
