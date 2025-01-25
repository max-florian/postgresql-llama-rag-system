import psycopg2
from flask import g
import os

DB_URL = os.getenv("DATABASE_URL")

def get_db():
    if "db" not in g:
        g.db = psycopg2.connect(DB_URL)
    return g.db

def get_cursor():
    db = get_db()
    return db.cursor()

def close_connection(exception=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()
