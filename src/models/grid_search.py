"""Recherche les meilleurs hyperparamètres d'un GradientBoostingRegressor par GridSearchCV.

Entrées : data/processed_data/X_train_scaled.csv, y_train.csv
Sortie  : models/best_params.pkl
"""

import pickle
from pathlib import Path

import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

DATA_DIR = Path("data/processed_data")
BEST_PARAMS_FILE = Path("models/best_params.pkl")


def main() -> None:
    params = yaml.safe_load(Path("params.yaml").read_text())["grid_search"]

    X_train = pd.read_csv(DATA_DIR / "X_train_scaled.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze("columns")

    grid = GridSearchCV(
        estimator=GradientBoostingRegressor(random_state=params["random_state"]),
        param_grid=params["param_grid"],
        cv=params["cv"],
        scoring=params["scoring"],
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)

    BEST_PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(BEST_PARAMS_FILE, "wb") as file:
        pickle.dump(grid.best_params_, file)
    print(f"Meilleurs paramètres : {grid.best_params_}")


if __name__ == "__main__":
    main()
