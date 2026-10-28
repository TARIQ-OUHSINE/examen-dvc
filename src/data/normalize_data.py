"""Normalise les variables explicatives avec un StandardScaler.

Le scaler est ajusté sur l'ensemble d'entraînement uniquement, puis appliqué aux deux ensembles.

Entrées : data/processed_data/X_train.csv, X_test.csv
Sorties : data/processed_data/X_train_scaled.csv, X_test_scaled.csv, models/scaler.pkl
"""

import pickle
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path("data/processed_data")
SCALER_FILE = Path("models/scaler.pkl")


def main() -> None:
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    X_train_scaled.to_csv(DATA_DIR / "X_train_scaled.csv", index=False)
    X_test_scaled.to_csv(DATA_DIR / "X_test_scaled.csv", index=False)

    SCALER_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SCALER_FILE, "wb") as file:
        pickle.dump(scaler, file)
    print("Normalisation terminée")


if __name__ == "__main__":
    main()
