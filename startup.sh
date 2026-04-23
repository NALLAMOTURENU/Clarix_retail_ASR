#!/bin/bash
# Azure App Service startup script
# App Service runs this via "Startup Command": bash startup.sh

python -c "from app import app, init_db; init_db(); print('DB ready.')"
gunicorn --bind=0.0.0.0:8000 --workers=2 --threads=4 --timeout=120 app:app
