"""Évalue le modèle entraîné sur l'ensemble de test.

Entrées : data/processed_data/X_test_scaled.csv, y_test.csv, models/gbr_model.pkl
Sorties : data/predictions.csv, metrics/scores.json
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = Path("data/processed_data")
MODEL_FILE = Path("models/gbr_model.pkl")
PREDICTIONS_FILE = Path("data/predictions.csv")
SCORES_FILE = Path("metrics/scores.json")


def main() -> None:
    X_test = pd.read_csv(DATA_DIR / "X_test_scaled.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze("columns")
    with open(MODEL_FILE, "rb") as file:
        model = pickle.load(file)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    scores = {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
    }

    pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).to_csv(PREDICTIONS_FILE, index=False)
    SCORES_FILE.parent.mkdir(parents=True, exist_ok=True)
    SCORES_FILE.write_text(json.dumps(scores, indent=4))
    print(f"Scores : {scores}")


if __name__ == "__main__":
    main()
