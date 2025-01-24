import os
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI, OpenAIError
from PyPDF2 import PdfReader
import time

# Load environment variables from .env file
load_dotenv()

# Set up OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set your OpenAI API key in the .env file.")

OpenAI.api_key = OPENAI_API_KEY

def pdf_to_embedding(pdf_path):
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

    # Step 2: Split text into smaller chunks for embedding
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Max tokens for ada embeddings is ~8196, this keeps it within limits
        chunk_overlap=200,  # Overlap between chunks to ensure context continuity
    )
    texts = text_splitter.create_documents(documents)

    # Step 3: Generate embeddings with retry logic
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)
    text_embeddings = []

    for text in texts:
        success = False
        while not success:
            try:
                embedding = embeddings.embed_query(text.page_content)
                text_embeddings.append(embedding)
                success = True
            except OpenAIError as e:
                print("OpenAI error:", e)
                time.sleep(5)  # Wait before retrying

    return text_embeddings

if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Transform a PDF into embeddings using LangChain and OpenAI.")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file to process.")
    args = parser.parse_args()

    if not os.path.exists(args.pdf_path):
        print(f"Error: File '{args.pdf_path}' not found.")
        exit(1)

    print("Processing PDF...")
    embeddings = pdf_to_embedding(args.pdf_path)
    print(f"Generated {len(embeddings)} embeddings from the PDF.")

    # Optionally save embeddings to a file
    output_path = args.pdf_path + ".embeddings.json"
    with open(output_path, "w") as f:
        import json
        json.dump(embeddings, f)

    print(f"Embeddings saved to {output_path}.")
