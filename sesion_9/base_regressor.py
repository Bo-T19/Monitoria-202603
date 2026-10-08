"""
base_regressor.py
=================
Clase base que compartirán todos los modelos de regresión.
Centraliza las funciones de: split, pipeline, métricas, predict, report y demás.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")


class BaseRegressor:
    """
    Clase base para todos los modelos de regresión.

    Gestiona:
    - La entrada de inputs
    - Train y test split
    - Construcción del pipeline
    - Cálculo de métricas
    """

    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        scale_features: bool = True,
    ):
        self.test_size = test_size
        self.random_state = random_state
        self.scale_features = scale_features

        # Inicialización de atributos
        self.pipeline_: Pipeline | None = None
        self.feature_names_: list[str] | None = None
        self.metrics_: dict = {}
        self.X_train_: np.ndarray | None = None
        self.X_test_: np.ndarray | None = None
        self.y_train_: np.ndarray | None = None
        self.y_test_: np.ndarray | None = None
        
    # Conversión

    def _names(self, X) -> list[str]:
        if isinstance(X, pd.DataFrame):
            return X.columns.tolist()
        n = X.shape[1] if hasattr(X, "shape") else len(X[0])
        return [f"x{i}" for i in range(n)]

    def _arr(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X.to_numpy()
        return np.array(X)

    def _arr1d(self, y) -> np.ndarray:
        if isinstance(y, pd.Series):
            return y.to_numpy()
        return np.array(y).ravel()

    # Split de datos

    def _split(self, X: np.ndarray, y: np.ndarray):
        (self.X_train_, self.X_test_, 
        self.y_train_, self.y_test_)= train_test_split(
        X, y, 
        test_size=self.test_size, 
        random_state=self.random_state
        )


    # Pipeline

    def _make_pipeline(self, model) -> Pipeline:
        steps = []
        if self.scale_features:
            steps.append(("scaler", StandardScaler()))
        steps.append(("model", model))
        return Pipeline(steps)

    # Métricas

    def _store_metrics(self, y_true: np.ndarray, y_pred: np.ndarray):
        mse = mean_squared_error(y_true, y_pred)

        self.metrics_ = {
            "MSE": round(mse, 6),
            "RMSE": round(np.sqrt(mse), 6),
            "MAE": round(mean_absolute_error(y_true, y_pred), 6),
            "R^2": round(r2_score(y_true, y_pred), 6),
        }


    # Funciones 
    
    def _check_fitted(self):
        if self.pipeline_ is None:
            raise RuntimeError("El modelo no ha sido entrenado aún. Llama a 'fit' primero.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones usando el pipeline entrenado.
        """
        self._check_fitted()
        return self.pipeline_.predict(self._arr(X))

    def get_metrics(self) -> dict:
        """
        Devuelve las métricas calculadas después del test.
        """
        self._check_fitted()
        return self.metrics_.copy()

    def print_metrics(self, title: str = "Métricas en test"):
        """
        Imprime un reporte con las métricas del modelo.
        """
        self._check_fitted()
        print(f"{title}:")
        print("-" * len(title))
        for metric, value in self.metrics_.items():
            print(f"{metric}: {value}")
        print("-" * len(title))
    
    def summary(self):
        """
        Devolver las métricas como un Dataframe para un análisis más detallado.
        """
        self._check_fitted()
        return pd.DataFrame([self.metrics_])