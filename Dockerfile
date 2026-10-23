# Image unique : application Streamlit (par défaut) ou API FastAPI (SERVICE=api).
# Le modèle est entraîné pendant le build : le conteneur démarre prêt.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
# libgomp1 : requis par LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/fraude
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
ARG FRAUD_N_CARDS=400
ENV FRAUD_N_CARDS=${FRAUD_N_CARDS} SERVICE=app
RUN python -m fraud.pipeline

EXPOSE 8501 8000
CMD ["sh", "docker/start.sh"]
