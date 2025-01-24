import os
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def pdf_to_embedding_hf(pdf_path):
    # Step 1: Load PDF using PyPDF2
    reader = PdfReader(pdf_path)
    documents = []

    # Extract text from each page
    for page in reader.pages:
        text = page.extract_text()
        if text:
            documents.append(text)

    # Save extracted text to a file
    text_output_path = pdf_path + ".extracted_text.txt"
    with open(text_output_path, "w") as text_file:
        for document in documents:
            text_file.write(document + "\n")

    print(f"Extracted text saved to {text_output_path}.")

    # Step 2: Generate embeddings using Hugging Face's SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Example model
    text_embeddings = model.encode(documents, convert_to_tensor=False)

    return documents, text_embeddings

def insert_into_db(documents, embeddings):
    # Get the PostgreSQL connection string from the .env file
    db_connection_string = os.getenv("DATABASE_URL")
    if not db_connection_string:
        raise ValueError("Please set your PostgreSQL connection string in the .env file as DATABASE_URL.")

    # Connect to the PostgreSQL database
    conn = psycopg2.connect(db_connection_string)
    cursor = conn.cursor()

    # Prepare data for insertion
    data = [(doc, embedding.tolist()) for doc, embedding in zip(documents, embeddings)]

    # Insert data into the "doc" table
    insert_query = """
    INSERT INTO doc (doc_text, embedding) VALUES %s
    """
    execute_values(cursor, insert_query, data)

    # Commit and close the connection
    conn.commit()
    cursor.close()
    conn.close()

    print("Data successfully inserted into the database.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Transform a PDF into embeddings and store in PostgreSQL using Hugging Face.")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file to process.")
    args = parser.parse_args()

    if not os.path.exists(args.pdf_path):
        print(f"Error: File '{args.pdf_path}' not found.")
        exit(1)

    print("Processing PDF...")
    documents, embeddings = pdf_to_embedding_hf(args.pdf_path)
    print(f"Generated embeddings for {len(embeddings)} chunks from the PDF.")

    print("Inserting data into the database...")
    insert_into_db(documents, embeddings)

    print("Process completed.")
