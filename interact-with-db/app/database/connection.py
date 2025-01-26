import psycopg2
from flask import g
import os

DB_URL = os.getenv("DATABASE_URL")

def get_db():
    if "db" not in g:
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            raise ValueError("DATABASE_URL environment variable is not set")
        g.db = psycopg2.connect(db_url)
    return g.db

def get_cursor():
    db = get_db()
    return db.cursor()

def close_connection(exception=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()
