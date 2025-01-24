CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE doc (
    doc_text TEXT NOT NULL,
    embedding VECTOR(384)
);