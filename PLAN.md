# PLAN — Application web de détection de fraude explicable

Projet Challenge 2 (Data Mining, ENSI 2026-2027). Sources : `Projet_Challenge_2_Detetion de Fraude_Explicable.pdf`
(cahier des charges) et `Ch7_Detection_de_fraude.pdf` (cours). `claude.md` est vide : aucune consigne
supplémentaire.

## 1. Choix du jeu de données

**Jeu n°1 du catalogue : Credit Card Transactions Fraud Detection (simulateur Sparkov)** — cible `is_fraud`.
C'est le jeu pour lequel le kit est configuré, et il contient tout ce qu'exige le cahier des charges :
identifiant de carte (`cc_num`), horodatage, marchand, catégorie, localisations porteur / marchand,
montant → variables de vélocité, de distance et d'écart au profil.

Trois modes de chargement (Module 1) :

| Mode | Usage |
|------|-------|
| Téléchargement automatique Kaggle (`kagglehub`, `kartik2112/fraud-detection`) | données réelles du simulateur, `fraudTrain.csv` + `fraudTest.csv` concaténés et triés par date |
| Import de fichier CSV (même schéma) | depuis l'interface ou l'API Python |
| Générateur synthétique au **même schéma** | démo hors ligne, tests automatisés, environnements sans accès Kaggle |

Pour les gros volumes, l'échantillonnage se fait **par carte** (et non par ligne) afin de conserver
l'historique complet de chaque porteur, indispensable aux variables comportementales.

## 2. Architecture

```
fraud/                 bibliothèque (logique métier, testée)
  config.py            constantes : colonnes, chemins, coûts par défaut
  synthetic.py         générateur Sparkov-like
  data.py              M1 chargement + statistiques exploratoires
  features.py          M2 variables, séparation chronologique, préprocesseur
  imbalance.py         M2 stratégies de déséquilibre + correction de probabilité
  models.py            M3 modèles supervisés et non supervisés
  evaluation.py        M4 métriques, precision@k, coût, seuil
  explain.py           M5 SHAP global / local, codes motifs
  pipeline.py          orchestration entraînement → artefacts (artifacts/)
app/streamlit_app.py   interface interactive (une page par module)
api/main.py            M6 API FastAPI
tests/                 pytest, un fichier par module
```

Stack : pandas, scikit-learn, LightGBM, imbalanced-learn, SHAP, Streamlit + Plotly, FastAPI.

## 3. Modules

### Module 1 — Données
- `load_kaggle()`, `load_csv()`, `generate_synthetic()` → DataFrame normalisé (`trans_date_trans_time` en datetime, tri chronologique).
- Statistiques : taux de fraude global, distribution des montants par classe, taux de fraude par mois/semaine,
  par heure, par catégorie, par jour de semaine, par genre / tranche d'âge.
- Page Streamlit « Données » : choix de la source, KPIs, graphiques Plotly.

### Module 2 — Préparation et variables
- Temporelles : `hour_sin`, `hour_cos` (codage cyclique), `dow` (+ codage cyclique), `is_night`.
- Comportementales par carte, calculées **uniquement sur le passé** (fenêtres fermées à gauche, la
  transaction courante est exclue) : `n_tx_1h`, `n_tx_24h`, `amt_ratio_30d` (montant / moyenne des 30 jours
  précédents), `secs_since_prev` (log). Implémentation vectorisée par `searchsorted` sur une clé
  (carte, temps) → linéaire en mémoire, rapide sur 1,8 M lignes.
- Autres : `log_amt`, distance porteur–marchand (haversine), âge, `log_city_pop`, genre, catégorie (one-hot).
- Pas de fuite : variables causales ; préprocesseur (standardisation, one-hot) ajusté sur l'entraînement seul.
- Séparation **chronologique 60 / 20 / 20** (entraînement / validation / test) par quantiles de temps ; repli sur
  une séparation aléatoire stratifiée, signalée, seulement si aucune colonne temporelle n'existe.
- Déséquilibre (entraînement seul) : comparaison en validation (PR-AUC) de *aucune*, *pondération des classes*,
  *sous-échantillonnage 1:5 / 1:10 / 1:20*, *SMOTE*. Correction bayésienne des probabilités après
  sous-échantillonnage : `p = β·ps / (β·ps − ps + 1)`. La stratégie gagnante est retenue automatiquement.

### Module 3 — Modélisation
- Supervisés : régression logistique pondérée (référence lisible) et LightGBM (modèle principal).
- Non supervisés (entraînés sans étiquettes, sur un échantillon de l'entraînement, variables standardisées) :
  **Isolation Forest**, **LOF** (`novelty=True`), **autoencodeur** (MLPRegressor à goulot, entraîné sur les
  transactions normales, score = erreur de reconstruction).
- Tableau comparatif de toutes les méthodes (validation et test).

### Module 4 — Évaluation et seuil
- PR-AUC, ROC-AUC, précision@k et rappel@k ; budget `k` = capacité d'analyse (alertes/jour, défaut 50)
  × nombre de jours de la période — justification : capacité d'une équipe d'analystes, ≈ quelques fois le
  nombre de fraudes attendu par jour.
- Au seuil retenu : F2, MCC, précision, rappel, matrice de confusion.
- Courbes précision-rappel superposées pour toutes les méthodes.
- Coût métier `C(τ) = Σ montants des fraudes manquées + Ca × N_alertes(τ)` ; τ minimisé sur la validation,
  confirmé sur le test, comparé au coût « sans modèle ».
- Page Streamlit : curseurs de seuil et de budget, coût d'analyse `Ca` paramétrable, mise à jour en direct des
  alertes, précision, rappel et coût.

### Module 5 — Explainable AI
- Global : SHAP beeswarm (TreeExplainer sur LightGBM) + importance par permutation.
- Local : waterfall SHAP d'une alerte choisie + **codes motifs** : les 3 à 5 variables ayant le plus poussé le
  score vers la fraude, traduites en phrases lisibles par un analyste (« montant 7,8 × la moyenne 30 j »,
  « transaction à 2 h du matin », …).
- Page Streamlit : file d'alertes triée par score, sélection d'une alerte, explication.

### Module 6 — API FastAPI
- `GET /health`, `GET /model` (version, variables, seuil, métriques).
- `POST /predict` : une ou plusieurs transactions brutes (schéma Sparkov) + historique optionnel de la carte
  → score, décision d'alerte au seuil coût-optimal, codes motifs.
- `POST /explain` : contributions SHAP complètes d'une transaction.
- Validation Pydantic, tests avec `TestClient`.

## 4. Livraison
- Un commit par module (1 → 6), chacun avec code + page d'application + tests verts.
- Commandes :
  - `python -m fraud.pipeline --source synthetic` (ou `--source kaggle` / `--csv fichier.csv`) — entraîne et
    écrit `artifacts/`.
  - `streamlit run app/streamlit_app.py`
  - `uvicorn api.main:app --reload`
  - `pytest -q`
