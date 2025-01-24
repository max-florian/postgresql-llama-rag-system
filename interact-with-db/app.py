import os
import psycopg2
from openai import OpenAI
from flask import Flask, request, jsonify
from psycopg2.extras import execute_values
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI API key
client = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY"),
)

# PostgreSQL connection URL
DB_URL = os.getenv("DATABASE_URL")

# Initialize Flask app
app = Flask(__name__)

# Function to connect to the PostgreSQL database
def connect_to_db():
    return psycopg2.connect(DB_URL)

# Function to compute the OpenAI embedding
def get_embedding(text, model="text-embedding-ada-002"):
    response = client.embeddings.create(model=model, input=text)
    return response["data"][0]["embedding"]

# Function to query the most relevant document from the database
def query_relevant_docs(user_query, top_k=1):
    # Compute the embedding for the user query
    query_embedding = get_embedding(user_query)

    # Connect to the database
    conn = connect_to_db()
    cursor = conn.cursor()

    # Query the top-k most relevant documents
    sql = """
        SELECT doc_text, (embedding <=> %s) AS distance
        FROM doc
        ORDER BY distance
        LIMIT %s;
    """
    cursor.execute(sql, (query_embedding, top_k))
    results = cursor.fetchall()

    # Close the connection
    cursor.close()
    conn.close()

    return results

# Chatbot function
def chatbot_response(user_query):
    # Query the database for relevant documents
    relevant_docs = query_relevant_docs(user_query, top_k=1)

    if not relevant_docs:
        return "I'm sorry, I couldn't find any relevant information."

    # Extract the most relevant document text
    doc_text = relevant_docs[0][0]

    # Use OpenAI GPT to generate a response
    prompt = f"User query: {user_query}\nRelevant information: {doc_text}\nChatbot response:"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150
    )
    return response["choices"][0]["message"]["content"].strip()

# Endpoint to interact with the database
@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    user_query = data.get("query", "")

    if not user_query:
        return jsonify({"error": "Query is required."}), 400

    try:
        response = chatbot_response(user_query)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint to test OpenAI key communication
@app.route("/test-openai", methods=["GET"])
def test_openai():
    try:
        test_text = "Hello, OpenAI!"
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": test_text}
            ],
            max_tokens=10
        )
        return jsonify({"status": "success", "message": "OpenAI communication successful."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)