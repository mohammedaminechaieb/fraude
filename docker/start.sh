#!/bin/sh
# SERVICE=app (défaut) → Streamlit sur $PORT (8501) ; SERVICE=api → FastAPI sur $PORT (8000).
set -e
if [ "$SERVICE" = "api" ]; then
    exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8000}"
else
    exec streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port "${PORT:-8501}" \
        --server.headless true
fi
