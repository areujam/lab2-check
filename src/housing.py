import random
from typing import Dict
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.datasets import fetch_california_housing
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np

BASE_PIPELINE = [
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
]

MODELS: Dict[str, Dict] = {
    # From main: expanded grid for ElasticNet
    "simple_elastic": {
        "pipeline": Pipeline(BASE_PIPELINE + [("model", ElasticNet(max_iter=2000))]),
        "param_grid": {
            "model__alpha": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 0.5, 0.7, 1.0, 3.0, 10.0, 30.0],
            "model__l1_ratio": [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        },
    },
    # Existing degree-2 polynomial
    "poly_elastic": {
        "pipeline": Pipeline(
            BASE_PIPELINE
            + [
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("model", ElasticNet(max_iter=2000)),
            ]
        ),
        "param_grid": {
            "model__alpha": [1e-4, 1e-3, 1e-2, 0.1, 0.3, 0.5, 1.0, 3.0, 10.0],
            "model__l1_ratio": [0.0, 0.5, 1.0],
        },
    },
    # NEW: degree-3 polynomial (from add_third_poly)
    "poly3_elastic": {
        "pipeline": Pipeline(
            BASE_PIPELINE
            + [
                ("poly", PolynomialFeatures(degree=3, include_bias=False)),
                ("model", ElasticNet(max_iter=2000)),
            ]
        ),
        "param_grid": {
            "model__alpha": [1e-4, 1e-3, 1e-2, 0.1, 0.3, 0.5, 1.0, 3.0, 10.0],
            "model__l1_ratio": [0.0, 0.5, 1.0],
        },
    },
    "knn": {
        "pipeline": Pipeline(BASE_PIPELINE + [("model", KNeighborsRegressor())]),
        "param_grid": {
            "model__n_neighbors": [2, 5, 10, 20, 50],
            "model__p": [1, 2],
        },
    },
}

def set_seed(seed: int = 1):
    np.random.seed(seed)
    random.seed(seed)

def load_dataset(test_size: float = 0.2, random_state: int = 1):
    ds = fetch_california_housing(as_frame=True)
    df = ds.frame.copy()
    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def train(
    model: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: int = 5,
    random_state: int = 1,
):
    if model not in MODELS:
        raise ValueError(f"Model {model} is not defined in MODELS.")
    pipeline = MODELS[model]["pipeline"]
    param_grid = MODELS[model]["param_grid"]

    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        refit=True,
    )
    grid_search.fit(X_train, y_train)
    return grid_search

def eval(grid_search: GridSearchCV, X_test: pd.DataFrame, y_test: pd.Series):
    best = grid_search.best_estimator_
    y_pred = best.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {
        "estimator": best["model"],  # the model inside the pipeline
        "preds": y_pred,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }
