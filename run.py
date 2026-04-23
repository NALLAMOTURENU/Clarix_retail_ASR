"""
Initialises the database and starts the dev server.
Run with:  python run.py
"""
from app import app, init_db

if __name__ == '__main__':
    init_db()
    print("Database initialised.")
    print("Open http://localhost:8000 in your browser.")
    print("Register a new account, then go to Load Data → 'Load Default CSVs'.")
    app.run(debug=True, host='0.0.0.0', port=8000)
