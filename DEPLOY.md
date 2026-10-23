# Guide de déploiement

Deux composants : l'**application Streamlit** (interface) et l'**API FastAPI**. Recommandation :

| Composant | Où | Pourquoi |
|---|---|---|
| Application | Streamlit Community Cloud (gratuit) | fait pour Streamlit ; l'app demande ~700-950 Mo de RAM |
| API | Render (gratuit, 512 Mo) via `render.yaml` | build qui entraîne le modèle, démarrage prêt |
| Les deux | Docker (`Dockerfile`, `docker-compose.yml`) | n'importe quel serveur ou PC |

## 0. Mettre le code sur `main`

Streamlit Cloud et Render prennent par défaut la branche `main`. Fusionner la branche de travail
(pull request sur GitHub → *Merge*), ou choisir la branche `claude/eager-brahmagupta-evbhr0` dans chaque outil.

## 1. Application → Streamlit Community Cloud

1. https://share.streamlit.io → se connecter avec GitHub (autoriser l'accès au dépôt).
2. **Create app** → *Deploy a public app from GitHub* :
   - Repository : `mohammedaminechaieb/fraude`
   - Branch : `main`
   - Main file path : `app/streamlit_app.py`
   - App URL : au choix
   - *Advanced settings* → Python **3.12**
3. **Deploy**. L'installation prend quelques minutes.
4. Au premier affichage, l'application génère les données et entraîne les modèles toute seule
   (≈ 1 min, message « Premier lancement… »). Toutes les pages sont ensuite prêtes.

Options (*Settings → Secrets*, format TOML) :
```toml
FRAUD_N_CARDS = "250"        # jeu plus petit si l'app manque de mémoire
KAGGLE_USERNAME = "..."      # pour le mode Kaggle de la page 1
KAGGLE_KEY = "..."
```
Après une mise en veille, l'app ré-entraîne automatiquement au réveil.

## 2. API → Render (Blueprint)

1. https://render.com → se connecter avec GitHub.
2. **New → Blueprint** → choisir le dépôt `fraude` (branche `main`). Render lit `render.yaml`.
3. **Apply**. Le build installe les dépendances puis entraîne le modèle (quelques minutes).
4. Vérifier : `https://fraude-api.onrender.com/health` → `"model_loaded": true`,
   documentation interactive : `https://fraude-api.onrender.com/docs`.

Sans Blueprint (*New → Web Service*), saisir à la main :

| Champ | Valeur |
|---|---|
| Root Directory | *(vide)* |
| Runtime | Python 3 |
| Build Command | `pip install -r requirements.txt && python -m fraud.pipeline` |
| Start Command | `uvicorn api.main:app --host 0.0.0.0 --port $PORT` |
| Variable d'environnement | `PYTHON_VERSION` = `3.12.7` |

L'offre gratuite s'endort après 15 min sans requête : la première requête suivante prend ~1 min.

## 3. Docker (serveur personnel, VPS, autre hébergeur)

```bash
docker compose up --build
# application : http://localhost:8501   API : http://localhost:8000/docs
```
Une seule image ; `SERVICE=app` (défaut) ou `SERVICE=api` choisit le service, `PORT` le port.
Le modèle est entraîné au build (`--build-arg FRAUD_N_CARDS=250` pour un jeu plus petit) :

```bash
docker build -t fraude .
docker run -p 8501:8501 fraude                              # application
docker run -p 8000:8000 -e SERVICE=api -e PORT=8000 fraude  # API
```
Tout hébergeur acceptant un Dockerfile (Render *Docker*, Railway, Fly.io, Google Cloud Run, Hugging Face
Spaces *Docker*) fonctionne : il fournit `PORT`, il suffit d'ajouter `SERVICE=api` pour l'API.

## Variables d'environnement

| Variable | Défaut | Rôle |
|---|---|---|
| `FRAUD_N_CARDS` | 400 | taille du jeu synthétique (build et auto-entraînement) |
| `FRAUD_AUTOTRAIN` | 1 | `0` désactive l'auto-entraînement de l'application |
| `FRAUD_ARTIFACTS_DIR` | `artifacts/` | dossier du modèle entraîné |
| `FRAUD_BUNDLE` | `artifacts/bundle.joblib` | modèle chargé par l'API |
| `SERVICE`, `PORT` | `app`, 8501 / 8000 | image Docker uniquement |
| `KAGGLE_USERNAME`, `KAGGLE_KEY` | — | mode Kaggle |
