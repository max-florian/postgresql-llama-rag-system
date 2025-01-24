import os
import psycopg2
from flask import Flask, request, jsonify
from psycopg2.extras import execute_values
from dotenv import load_dotenv
import requests
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
import json

# Load environment variables
load_dotenv()

# PostgreSQL connection URL
DB_URL = os.getenv("DATABASE_URL")

# Ollama server configuration
OLLAMA_URL = "http://localhost:11434"  # Ollama server running on port 11434
MODEL_NAME = "llama3.2:1b"

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Establish a persistent database connection
conn = None
cursor = None

def setup_database_connection():
    global conn, cursor
    if conn is None or conn.closed:
        logging.info("Setting up persistent database connection.")
        conn = psycopg2.connect(DB_URL)
        cursor = conn.cursor()

@app.before_request
def before_request():
    setup_database_connection()

@app.teardown_appcontext
def teardown_database_connection(exception):
    global conn, cursor
    if cursor:
        cursor.close()
        logging.info("Database cursor closed.")
    if conn:
        conn.close()
        logging.info("Database connection closed.")

# Function to compute the embedding using SentenceTransformer
def get_embedding(text):
    logging.info("Generating embedding using SentenceTransformer.")
    try:
        embedding = embedding_model.encode(text).tolist()  # Convert to list for compatibility
        logging.info(f"Embedding generated successfully with dimension: {len(embedding)}")
        return embedding
    except Exception as e:
        logging.error(f"Error generating embedding: {e}")
        raise Exception(f"Error generating embedding: {e}")

# Function to query the most relevant document from the database
def query_relevant_docs(user_query, top_k=1):
    logging.info("Starting query_relevant_docs function.")

    try:
        logging.info("Computing embedding for user query.")
        query_embedding = get_embedding(user_query)
        logging.info("Embedding for user query computed successfully.")
    except Exception as e:
        logging.error(f"Failed to compute embedding: {e}")
        raise

    sql = """
        SELECT doc_text, (embedding <=> %s::vector) AS distance
        FROM doc
        ORDER BY distance
        LIMIT %s;
    """
    try:
        logging.info("Preparing and executing database query for relevant documents.")
        cursor.execute(sql, (query_embedding, top_k))
        results = cursor.fetchall()
        logging.info(f"Database query executed successfully. Retrieved {len(results)} results.")
    except Exception as e:
        logging.error(f"Error during database query execution: {e}")
        raise

    return results

# Chatbot function
def chatbot_response(user_query):
    logging.info("Querying relevant documents.")
    relevant_docs = query_relevant_docs(user_query, top_k=1)

    if not relevant_docs:
        logging.warning("No relevant documents found.")
        return "I'm sorry, I couldn't find any relevant information."

    logging.info("Generating response using Ollama.")
    doc_text = relevant_docs[0][0]
    payload = {
        "model": MODEL_NAME,
        "prompt": f"This is the relevant information: {doc_text}. This is the user question: {user_query}. Generate a precise response"
    }
    response = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, stream=True)

    if response.status_code == 200:
        try:
            # Process streamed JSON response
            response_text = ""
            for line in response.iter_lines():
                if line:
                    try:
                        json_line = json.loads(line)
                        if "response" in json_line:
                            response_text += json_line["response"]
                    except json.JSONDecodeError as e:
                        logging.error(f"Error decoding JSON line: {e}")
                        continue

            logging.info("Successfully generated response from Ollama.")
            return response_text.strip()
        except Exception as e:
            logging.error(f"Error processing streamed response: {e}")
            raise Exception(f"Error processing streamed response: {e}")
    else:
        logging.error(f"Failed to get chatbot response: {response.text}")
        raise Exception(f"Failed to get chatbot response: {response.text}")


# Endpoint to interact with the database
@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    user_query = data.get("query", "")

    if not user_query:
        logging.warning("No query provided in the request.")
        return jsonify({"error": "Query is required."}), 400

    try:
        logging.info("Received query request. Processing...")
        response = chatbot_response(user_query)
        logging.info("Successfully processed query request.")
        return jsonify({"response": response})
    except Exception as e:
        logging.error(f"Error during query processing: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Endpoint to test Ollama server communication
@app.route("/test-ollama", methods=["GET"])
def test_ollama():
    try:
        logging.info("Testing Ollama server communication.")
        test_text = "Hello, Ollama!"
        payload = {
            "model": MODEL_NAME,
            "prompt": test_text
        }
        response = requests.post(f"{OLLAMA_URL}/api/generate", json=payload)
        if response.status_code == 200:
            logging.info("Ollama server communication successful.")
            return jsonify({"status": "success", "message": "Ollama communication successful."})
        else:
            logging.error(f"Ollama server communication failed: {response.text}")
            raise Exception(response.text)
    except Exception as e:
        logging.error(f"Error during Ollama server test: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
