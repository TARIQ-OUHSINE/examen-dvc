"""Découpe le jeu de données brut en ensembles d'entraînement et de test.

Entrée  : data/raw_data/raw.csv
Sorties : data/processed_data/X_train.csv, X_test.csv, y_train.csv, y_test.csv
"""

from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

RAW_FILE = Path("data/raw_data/raw.csv")
OUTPUT_DIR = Path("data/processed_data")


def main() -> None:
    params = yaml.safe_load(Path("params.yaml").read_text())["split"]

    df = pd.read_csv(RAW_FILE)
    # Seules les variables numériques sont utilisées (une éventuelle colonne de date est écartée)
    df = df.select_dtypes(include="number")

    X = df.drop(columns=params["target"])
    y = df[params["target"]]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["test_size"], random_state=params["random_state"]
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
    X_test.to_csv(OUTPUT_DIR / "X_test.csv", index=False)
    y_train.to_csv(OUTPUT_DIR / "y_train.csv", index=False)
    y_test.to_csv(OUTPUT_DIR / "y_test.csv", index=False)
    print(f"Split : {len(X_train)} lignes d'entraînement, {len(X_test)} lignes de test, "
          f"{X.shape[1]} variables explicatives")


if __name__ == "__main__":
    main()
