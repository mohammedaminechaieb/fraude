# Détection de fraude explicable

Projet Challenge 2 — Data Mining, ENSI 2026-2027. Application web interactive (Streamlit) et API (FastAPI)
qui classent des transactions par risque, fixent un seuil d'alerte selon un coût métier et expliquent chaque
alerte avec SHAP. Jeu de données : **n°1 du catalogue, Credit Card Transactions Fraud Detection (Sparkov)**.
Le plan détaillé et les choix sont dans [PLAN.md](PLAN.md).

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

```bash
# 1. entraîner (écrit artifacts/bundle.joblib)
python -m fraud.pipeline --source synthetic             # générateur hors ligne au schéma Sparkov
python -m fraud.pipeline --source kaggle --n-cards 300  # Kaggle (identifiants requis), échantillon de cartes
python -m fraud.pipeline --csv fraudTrain.csv           # fichier local

# 2. application web
streamlit run app/streamlit_app.py

# 3. API — documentation sur http://localhost:8000/docs
uvicorn api.main:app --reload

# tests
pytest -q
```

Le téléchargement Kaggle utilise `kagglehub` et nécessite `~/.kaggle/kaggle.json` ou les variables
`KAGGLE_USERNAME` / `KAGGLE_KEY`. Sans accès Kaggle, le générateur synthétique reproduit le schéma et les
mécanismes de fraude du simulateur (rafales nocturnes, catégories en ligne, montants élevés, part de fraudes
discrètes).

## Déploiement

Streamlit Community Cloud (application), Render (API, `render.yaml`) ou Docker (`docker compose up --build`) :
voir [DEPLOY.md](DEPLOY.md). Au premier lancement, l'application entraîne les modèles toute seule.

## Modules

| # | Module | Code | Page |
|---|--------|------|------|
| 1 | Données : chargement, taux de fraude, montants par classe, évolution temporelle, taux par heure / jour / catégorie / âge | `fraud/data.py`, `fraud/synthetic.py` | 1 · Données |
| 2 | Variables cycliques et comportementales sans fuite, séparation chronologique 60/20/20, comparaison des stratégies de déséquilibre | `fraud/features.py`, `fraud/imbalance.py` | 2 · Préparation |
| 3 | LightGBM, régression logistique pondérée, Isolation Forest, LOF, autoencodeur ; tableau comparatif | `fraud/models.py`, `fraud/pipeline.py` | 3 · Modélisation |
| 4 | PR-AUC, ROC-AUC, précision@k / rappel@k, F2, MCC, seuil au coût minimal, curseurs en direct | `fraud/evaluation.py` | 4 · Évaluation et seuil |
| 5 | SHAP beeswarm, importance par permutation, waterfall et codes motifs | `fraud/explain.py` | 5 · Explications |
| 6 | API FastAPI `/health`, `/model`, `/predict`, `/explain` | `api/main.py` | 6 · API |

## Résultats sur le jeu synthétique par défaut (400 cartes, 284 061 transactions, 0,44 % de fraudes)

Stratégie de déséquilibre retenue en validation : SMOTE (PR-AUC val 0,675 contre 0,625 sans rééchantillonnage).
Seuil LightGBM choisi au coût minimal en validation (Ca = 10 par alerte), confirmé sur le test (147 jours) :

| Méthode | PR-AUC | ROC-AUC | Alertes | Précision | Rappel | F2 | MCC | Coût |
|---|---|---|---|---|---|---|---|---|
| LightGBM | 0,797 | 0,968 | 603 | 0,338 | 0,833 | 0,644 | 0,528 | 13 205 |
| Régression logistique | 0,510 | 0,956 | 653 | 0,248 | 0,661 | 0,496 | 0,401 | 24 276 |
| LOF | 0,270 | 0,926 | 1 300 | 0,118 | 0,624 | 0,336 | 0,265 | 38 263 |
| Isolation Forest | 0,368 | 0,900 | 623 | 0,196 | 0,498 | 0,381 | 0,308 | 45 938 |
| Autoencodeur | 0,150 | 0,843 | 802 | 0,115 | 0,376 | 0,258 | 0,202 | 77 461 |

Coût sans modèle (toutes les fraudes passent) : 118 680. Budget : 5 alertes/jour (≈ 3 × les fraudes
quotidiennes de l'entraînement), soit k = 735 sur le test : précision@k 0,279, rappel@k 0,837.
Ces chiffres portent sur des données simulées et ne préjugent pas des performances sur le vrai jeu Kaggle.
