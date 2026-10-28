"""Entraîne le GradientBoostingRegressor avec les meilleurs paramètres du GridSearch.

Entrées : data/processed_data/X_train_scaled.csv, y_train.csv, models/best_params.pkl
Sortie  : models/gbr_model.pkl
"""

import pickle
from pathlib import Path

import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingRegressor

DATA_DIR = Path("data/processed_data")
BEST_PARAMS_FILE = Path("models/best_params.pkl")
MODEL_FILE = Path("models/gbr_model.pkl")


def main() -> None:
    random_state = yaml.safe_load(Path("params.yaml").read_text())["grid_search"]["random_state"]

    X_train = pd.read_csv(DATA_DIR / "X_train_scaled.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze("columns")
    with open(BEST_PARAMS_FILE, "rb") as file:
        best_params = pickle.load(file)

    model = GradientBoostingRegressor(**best_params, random_state=random_state)
    model.fit(X_train, y_train)

    with open(MODEL_FILE, "wb") as file:
        pickle.dump(model, file)
    print(f"Modèle entraîné et sauvegardé dans {MODEL_FILE}")


if __name__ == "__main__":
    main()
