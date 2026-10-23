"""Générateur de transactions au schéma Sparkov.

Sert de source hors ligne (démo, tests, environnements sans accès Kaggle). Les mécanismes de
fraude imitent ceux du simulateur Sparkov : une carte est compromise à une date aléatoire et
subit une rafale de transactions frauduleuses sur quelques jours, plutôt la nuit, sur des
catégories « en ligne » et avec des montants élevés. Une part des fraudes est volontairement
discrète (montant et horaire ordinaires) pour que le problème ne soit pas trivial.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SPARKOV_COLUMNS

# catégorie -> (poids global, montant médian légitime, sigma log-normal)
CATEGORIES = {
    "gas_transport": (0.10, 60, 0.5),
    "grocery_pos": (0.10, 100, 0.5),
    "home": (0.09, 55, 0.9),
    "shopping_pos": (0.09, 70, 1.0),
    "kids_pets": (0.09, 45, 0.9),
    "shopping_net": (0.07, 60, 1.0),
    "entertainment": (0.07, 50, 0.9),
    "food_dining": (0.07, 40, 0.9),
    "personal_care": (0.07, 35, 0.9),
    "health_fitness": (0.07, 45, 0.8),
    "misc_pos": (0.06, 45, 1.0),
    "misc_net": (0.05, 55, 1.1),
    "grocery_net": (0.035, 50, 0.6),
    "travel": (0.035, 90, 1.4),
}
# catégorie -> (poids, montant médian frauduleux, sigma)
FRAUD_CATEGORIES = {
    "shopping_net": (0.25, 950, 0.25),
    "grocery_pos": (0.24, 310, 0.15),
    "misc_net": (0.13, 780, 0.3),
    "shopping_pos": (0.12, 850, 0.3),
    "gas_transport": (0.10, 12, 0.3),
    "misc_pos": (0.06, 220, 0.6),
    "home": (0.04, 250, 0.6),
    "entertainment": (0.03, 500, 0.4),
    "travel": (0.03, 10, 0.5),
}
CITIES = [
    ("Houston", "TX", 29.76, -95.37), ("Dallas", "TX", 32.78, -96.80), ("Phoenix", "AZ", 33.45, -112.07),
    ("Denver", "CO", 39.74, -104.99), ("Chicago", "IL", 41.88, -87.63), ("Detroit", "MI", 42.33, -83.05),
    ("Atlanta", "GA", 33.75, -84.39), ("Miami", "FL", 25.76, -80.19), ("Seattle", "WA", 47.61, -122.33),
    ("Portland", "OR", 45.52, -122.68), ("Boston", "MA", 42.36, -71.06), ("Columbus", "OH", 39.96, -83.0),
    ("Nashville", "TN", 36.16, -86.78), ("Omaha", "NE", 41.26, -95.93), ("Albany", "NY", 42.65, -73.75),
    ("Tulsa", "OK", 36.15, -95.99), ("Boise", "ID", 43.62, -116.2), ("Fargo", "ND", 46.88, -96.79),
    ("Reno", "NV", 39.53, -119.81), ("Mobile", "AL", 30.69, -88.04),
]
FIRST = ["James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda", "William",
         "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica", "Thomas", "Sarah"]
LAST = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez",
        "Martinez", "Hernandez", "Lopez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson"]
JOBS = ["Engineer", "Teacher", "Nurse", "Accountant", "Lawyer", "Designer", "Electrician", "Pharmacist",
        "Surveyor", "Chef", "Journalist", "Architect", "Retail manager", "IT consultant"]
MERCHANT_STEMS = ["Kuhn", "Boyer", "Schultz", "Kozey", "Rau", "Heller", "Kling", "Bartoletti", "Hickle",
                  "Lind", "Stokes", "Gerlach", "Kautzer", "Ritchie", "Abbott", "Fisher", "Walter", "Koss"]

# distribution horaire des transactions légitimes (forte activité en journée)
_LEGIT_HOURS = np.array([1, 1, 1, 1, 1, 2, 4, 6, 8, 9, 10, 10, 12, 13, 13, 12, 12, 12, 11, 10, 9, 7, 4, 2],
                        dtype=float)
# fraudes : surtout entre 22 h et 3 h
_FRAUD_HOURS = np.array([10, 10, 10, 10, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 12, 12],
                        dtype=float)


def _merchant_location(rng, lat, lon, max_deg):
    return lat + rng.uniform(-max_deg, max_deg, len(lat)), lon + rng.uniform(-max_deg, max_deg, len(lon))


def generate_synthetic(
    n_cards: int = 400,
    start: str = "2019-01-01",
    end: str = "2020-12-31",
    mean_daily_rate: float = 1.0,
    compromised_share: float = 0.4,
    seed: int = 42,
) -> pd.DataFrame:
    """Génère un jeu de transactions au schéma Sparkov, trié chronologiquement."""
    rng = np.random.default_rng(seed)
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1)
    n_days = (t1 - t0).days

    cats = list(CATEGORIES)
    base_w = np.array([CATEGORIES[c][0] for c in cats])
    base_w /= base_w.sum()

    # --- profils des porteurs ---
    cards = pd.DataFrame({
        "cc_num": rng.integers(10**14, 10**16, n_cards, dtype=np.int64),
        "first": rng.choice(FIRST, n_cards),
        "last": rng.choice(LAST, n_cards),
        "gender": rng.choice(["F", "M"], n_cards),
        "job": rng.choice(JOBS, n_cards),
        "dob": t0 - pd.to_timedelta(rng.integers(18 * 365, 85 * 365, n_cards), unit="D"),
        "rate": rng.gamma(2.0, mean_daily_rate / 2.0, n_cards),
        "amt_mult": rng.lognormal(0, 0.35, n_cards),
    })
    city_idx = rng.integers(0, len(CITIES), n_cards)
    cards["city"] = [CITIES[i][0] for i in city_idx]
    cards["state"] = [CITIES[i][1] for i in city_idx]
    cards["lat"] = np.array([CITIES[i][2] for i in city_idx]) + rng.normal(0, 0.8, n_cards)
    cards["long"] = np.array([CITIES[i][3] for i in city_idx]) + rng.normal(0, 0.8, n_cards)
    cards["city_pop"] = np.round(rng.lognormal(9.0, 1.8, n_cards)).astype(int) + 100
    cards["zip"] = rng.integers(10000, 99999, n_cards)
    cards["street"] = [f"{n} {s} Street" for n, s in zip(rng.integers(1, 9999, n_cards), rng.choice(LAST, n_cards))]
    prefs = rng.dirichlet(base_w * 30, n_cards)

    # --- transactions légitimes ---
    n_tx = rng.poisson(cards["rate"].to_numpy() * n_days)
    owner = np.repeat(np.arange(n_cards), n_tx)
    total = len(owner)
    day = rng.integers(0, n_days, total)
    hour = rng.choice(24, total, p=_LEGIT_HOURS / _LEGIT_HOURS.sum())
    secs = rng.integers(0, 3600, total)
    u = rng.random(total)
    cum = np.cumsum(prefs, axis=1)[owner]
    cat_idx = (u[:, None] > cum).sum(axis=1).clip(max=len(cats) - 1)
    med = np.array([CATEGORIES[c][1] for c in cats])[cat_idx]
    sig = np.array([CATEGORIES[c][2] for c in cats])[cat_idx]
    amt = med * cards["amt_mult"].to_numpy()[owner] * rng.lognormal(0, sig)
    legit = pd.DataFrame({
        "card": owner,
        "ts": t0 + pd.to_timedelta(day * 86400 + hour * 3600 + secs, unit="s"),
        "category": np.array(cats)[cat_idx],
        "amt": amt,
        "dist": 0.9,
        "is_fraud": 0,
    })

    # --- épisodes de fraude ---
    fcats = list(FRAUD_CATEGORIES)
    fw = np.array([FRAUD_CATEGORIES[c][0] for c in fcats])
    fw /= fw.sum()
    compromised = rng.choice(n_cards, int(compromised_share * n_cards), replace=False)
    rows = []
    for c in compromised:
        k = rng.integers(4, 13)
        start_s = rng.integers(0, (n_days - 5) * 86400)
        span = rng.integers(1, 4) * 86400
        stealth = rng.random() < 0.2  # fraudeur discret : imite le comportement du porteur
        offs = np.sort(rng.integers(0, span, k))
        days_ = (start_s + offs) // 86400
        if stealth:
            hrs = rng.choice(24, k, p=_LEGIT_HOURS / _LEGIT_HOURS.sum())
            fc = rng.choice(cats, k, p=prefs[c])
            fa = np.array([CATEGORIES[x][1] for x in fc]) * cards["amt_mult"].iat[c] * rng.lognormal(0.5, 0.6, k)
            dist = 1.0
        else:
            hrs = rng.choice(24, k, p=_FRAUD_HOURS / _FRAUD_HOURS.sum())
            fc = rng.choice(fcats, k, p=fw)
            fa = np.array([FRAUD_CATEGORIES[x][1] for x in fc]) * rng.lognormal(0, [FRAUD_CATEGORIES[x][2] for x in fc])
            dist = 1.6
        for d, h, cat, a in zip(days_, hrs, fc, fa):
            rows.append((c, t0 + pd.Timedelta(seconds=int(d * 86400 + h * 3600 + rng.integers(0, 3600))),
                         cat, a, dist, 1))
    fraud = pd.DataFrame(rows, columns=legit.columns)
    tx = pd.concat([legit, fraud], ignore_index=True)

    # quelques légitimes atypiques (achats nocturnes ou gros montants) pour brouiller les signaux simples
    odd = rng.random(len(tx)) < 0.01
    tx.loc[odd & (tx.is_fraud == 0), "amt"] *= rng.uniform(3, 12, (odd & (tx.is_fraud == 0)).sum())

    # --- habillage au schéma Sparkov ---
    cd = cards.iloc[tx["card"].to_numpy()].reset_index(drop=True)
    out = pd.DataFrame({
        "trans_date_trans_time": tx["ts"].to_numpy(),
        "cc_num": cd["cc_num"].to_numpy(),
        "category": tx["category"].to_numpy(),
        "amt": np.round(tx["amt"].to_numpy().clip(1.0, 30000), 2),
    })
    stems = rng.choice(MERCHANT_STEMS, len(out))
    out["merchant"] = ["fraud_" + s + "-" + c.split("_")[0].title() for s, c in zip(stems, out["category"])]
    for col in ["first", "last", "gender", "street", "city", "state", "zip", "lat", "long", "city_pop", "job", "dob"]:
        out[col] = cd[col].to_numpy()
    out["merch_lat"], out["merch_long"] = _merchant_location(
        rng, out["lat"].to_numpy(), out["long"].to_numpy(), 1.0)
    far = tx["dist"].to_numpy()
    out["merch_lat"] = out["lat"] + (out["merch_lat"] - out["lat"]) * far
    out["merch_long"] = out["long"] + (out["merch_long"] - out["long"]) * far
    out["is_fraud"] = tx["is_fraud"].to_numpy().astype(int)
    out = out.sort_values("trans_date_trans_time", kind="stable").reset_index(drop=True)
    out["trans_num"] = [f"{x:032x}" for x in rng.integers(0, 2**62, len(out), dtype=np.int64)]
    out["unix_time"] = (out["trans_date_trans_time"] - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)
    return out[SPARKOV_COLUMNS]
