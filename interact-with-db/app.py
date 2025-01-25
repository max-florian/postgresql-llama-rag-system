from app import create_app
from flask import g

app = create_app()

@app.teardown_appcontext
def teardown(exception=None):
    from app.database.connection import close_connection
    close_connection(exception)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
