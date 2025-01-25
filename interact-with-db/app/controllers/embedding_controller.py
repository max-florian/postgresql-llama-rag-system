from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text):
    embedding = embedding_model.encode(text).tolist()
    return embedding
