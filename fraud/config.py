"""Constantes partagées : schéma Sparkov, chemins et paramètres métier par défaut."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"
DATA_DIR = ROOT / "data"

# Schéma du simulateur Sparkov (Kaggle « kartik2112/fraud-detection »)
TIME_COL = "trans_date_trans_time"
CARD_COL = "cc_num"
AMOUNT_COL = "amt"
TARGET = "is_fraud"
CATEGORY_COL = "category"
REQUIRED_COLUMNS = [CARD_COL, CATEGORY_COL, AMOUNT_COL, TARGET]
SPARKOV_COLUMNS = [
    TIME_COL, CARD_COL, "merchant", CATEGORY_COL, AMOUNT_COL, "first", "last", "gender",
    "street", "city", "state", "zip", "lat", "long", "city_pop", "job", "dob", "trans_num",
    "unix_time", "merch_lat", "merch_long", TARGET,
]
KAGGLE_DATASET = "kartik2112/fraud-detection"

# Paramètres métier par défaut (montants en unités monétaires du jeu)
DEFAULT_ALERT_COST = 10.0        # coût d'analyse d'une alerte (≈ 15 min d'analyste)
DEFAULT_DAILY_BUDGET = 50        # alertes qu'une équipe peut examiner par jour
SPLIT_FRACTIONS = (0.6, 0.2, 0.2)
RANDOM_STATE = 42
