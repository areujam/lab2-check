from __future__ import annotations
import numpy as np, pandas as pd
from typing import Dict, Any, Optional
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor

RANDOM_SEED = 1

def set_seed(seed: int = 1) -> None:
    global RANDOM_SEED
    RANDOM_SEED = int(seed)
    np.random.seed(RANDOM_SEED)

def load_dataset(test_size: float = 0.2, random_state: Optional[int] = None):
    random_state = RANDOM_SEED if random_state is None else random_state
    data = fetch_california_housing(as_frame=True)
    X = data.frame.drop(columns=["MedHouseVal"])
    y = data.frame["MedHouseVal"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def _numeric_preprocessor():
    return Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("scaler", StandardScaler())])

NUMERIC_TRANSFORMER = _numeric_preprocessor()

MODELS: Dict[str, Dict[str, Any]] = {
    "simple_elastic": {
        "pipeline": Pipeline([("pre", NUMERIC_TRANSFORMER),
                              ("model", ElasticNet(max_iter=10000, random_state=RANDOM_SEED))]),
        "param_grid": {"model__alpha": [0.0005, 0.001, 0.01, 0.1, 1.0],
                       "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]},
    },
    "poly_elastic": {
        "pipeline": Pipeline([("poly", PolynomialFeatures(degree=2, include_bias=False)),
                              ("pre", StandardScaler(with_mean=False)),
                              ("model", ElasticNet(max_iter=10000, random_state=RANDOM_SEED))]),
        "param_grid": {"model__alpha": [0.001, 0.01, 0.1],
                       "model__l1_ratio": [0.1, 0.5, 0.9],
                       "poly__degree": [2]},
    },
    "knn": {
        "pipeline": Pipeline([("pre", NUMERIC_TRANSFORMER),
                              ("model", KNeighborsRegressor())]),
        "param_grid": {"model__n_neighbors": [3, 5, 7, 11],
                       "model__weights": ["uniform", "distance"],
                       "model__p": [1, 2]},
    },
}

def train(model: str, X_train: pd.DataFrame, y_train: pd.Series,
          cv_splits: int = 5, random_state: Optional[int] = None, n_jobs: int = -1) -> GridSearchCV:
    if model not in MODELS:
        raise ValueError(f"Unknown model '{model}'. Choose from: {list(MODELS.keys())}")
    random_state = RANDOM_SEED if random_state is None else random_state
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    pipe = MODELS[model]["pipeline"]
    grid = MODELS[model]["param_grid"]
    search = GridSearchCV(pipe, grid, cv=cv, scoring="neg_mean_squared_error", n_jobs=n_jobs, refit=True)
    search.fit(X_train, y_train)
    return search

def eval(search: GridSearchCV, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    y_pred = search.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return {"estimator": search.best_estimator_,
            "best_params": search.best_params_,
            "y_pred": y_pred,
            "rmse": float(np.sqrt(mse)),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "r2": float(r2_score(y_test, y_pred))}
