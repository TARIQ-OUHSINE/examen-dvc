# Examen DVC et DagsHub

Workflow de modélisation de la concentration de silice (`silica_concentrate`) dans un processus de flottation,
versionné avec DVC et stocké sur DagsHub.

Données : https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv

## Pipeline

```txt
data/raw_data/raw.csv.dvc -> split -> normalize -> grid_search -> train -> evaluate
```

| Étape | Script | Sorties |
|---|---|---|
| `split` | `src/data/split_data.py` | `data/processed_data/X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv` |
| `normalize` | `src/data/normalize_data.py` | `data/processed_data/X_train_scaled.csv`, `X_test_scaled.csv`, `models/scaler.pkl` |
| `grid_search` | `src/models/grid_search.py` | `models/best_params.pkl` |
| `train` | `src/models/train_model.py` | `models/gbr_model.pkl` |
| `evaluate` | `src/models/evaluate_model.py` | `data/predictions.csv`, `metrics/scores.json` |

Les étapes et leurs dépendances sont décrites dans `dvc.yaml`. Les paramètres (taille du jeu de test,
grille d'hyperparamètres, etc.) sont dans `params.yaml`.

## Choix effectués

- **Modèle** : `GradientBoostingRegressor`, adapté à des relations non linéaires entre les paramètres
  opérationnels et la concentration de silice.
- **Normalisation** : `StandardScaler` ajusté sur l'ensemble d'entraînement seulement, puis appliqué au test,
  pour éviter toute fuite d'information. Le scaler est sauvegardé pour normaliser de nouvelles données.
- **GridSearch** : validation croisée à 5 plis sur `n_estimators`, `learning_rate` et `max_depth`,
  en minimisant l'erreur quadratique moyenne.
- **Métriques** : MSE, RMSE, MAE et R² dans `metrics/scores.json`, suivies par DVC (`dvc metrics show`).

## Reproduire

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
dvc pull                         # récupère les données et les modèles depuis DagsHub
dvc repro                        # relance les étapes dont les entrées ont changé
dvc metrics show
```
