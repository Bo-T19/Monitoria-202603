"""
random_forest_regression.py
===========================
Implementación del modelo de regresión con árboles aleatorios.
Opcionalmente puede hacer búsqueda de hiperparámetros con GridSearchCV o con RandomizedSearchCV.

"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from base_regressor import BaseRegressor

class RandomForestRegression(BaseRegressor):
    """
    Regresión con Random Forest.

    Parámetros:
    ----------
    n_estimators: int, default=100
        Número de árboles en el bosque.
    max_depth: int, default=None
        Profundidad máxima de los árboles. Si es None, se permiten nodos terminales.
    tune: bool, default=False
        Si es True, realiza búsqueda de hiperparámetros con GridSearchCV o RandomizedSearchCV.
    search_type: {'random', 'grid'}, default='random'
        Tipo de búsqueda de hiperparámetros: 'grid' o 'random'.
    param_grid: dict, default=None
        Diccionario con los hiperparámetros a buscar. Si es None, se usan valores por defecto.
    n_iter: int, default=20
        Número de iteraciones para RandomizedSearchCV. Solo se usa si search_type='random'.
    cv: int, default=5
        Número de folds para la validación cruzada.
    test_size: float, default=0.2
        Proporción del conjunto de datos que se usará para pruebas.
    random_state: int, default=42
        Semilla para la generación de números aleatorios.
    scale_features: bool, default=True
        Si es True, escala las características usando StandardScaler.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        tune: bool = False,
        search_type: str = "random",
        param_grid: dict | None = None,
        n_iter: int = 20,
        cv: int = 5,
        test_size: float = 0.2,
        random_state: int = 42,
        scale_features: bool = True,
    ):
        super().__init__(test_size=test_size, random_state=random_state, scale_features=scale_features)
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.tune = tune
        self.search_type = search_type
        self.param_grid = param_grid
        self.n_iter = n_iter
        self.cv = cv
        self.best_params_ : dict | None = None

    def _default_grid(self) -> dict:
        """
        Devuelve un diccionario con los hiperparámetros por defecto para la búsqueda.
        """
        return {
            "model__n_estimators":      [50, 100, 200, 300],
            "model__max_depth":         [None, 5, 10, 20],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf":  [1, 2, 4],
            "model__max_features":      ["sqrt", "log2", 0.5],
        }

    # Funciones públicas

    def fit(self, X, y) -> "RandomForestRegression":
        """
        Ajusta el modelo a los datos de entrenamiento.

        Parámetros:
        ----------
        X: array-like, shape (n_samples, n_features)
            Datos de entrada.
        y: array-like, shape (n_samples,)
            Valores objetivo.

        Devuelve:
        -------
        self: RandomForestRegression
            El objeto ajustado.
        """
        # Conversión de datos
        X = self._arr(X)
        y = self._arr1d(y)
        self.feature_names_ = self._names(X)

        # Split de datos
        self._split(X, y)

        rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )

        pipeline = self._make_pipeline(rf)

        if self.tune:
            if self.param_grid is None:
                self.param_grid = self._default_grid()

            if self.search_type == "grid":
                search = GridSearchCV(
                    pipeline,
                    param_grid=self.param_grid,
                    cv=self.cv,
                    n_jobs=-1,
                    scoring="neg_mean_squared_error",
                )
            elif self.search_type == "random":
                search = RandomizedSearchCV(
                    pipeline,
                    param_distributions=self.param_grid,
                    n_iter=self.n_iter,
                    cv=self.cv,
                    n_jobs=-1,
                    scoring="neg_mean_squared_error",
                    random_state=self.random_state,
                )

            search.fit(self.X_train_, self.y_train_)
            self.pipeline_ = search.best_estimator_
            self.best_params_ = search.best_params_
        else:
            self.pipeline_ = pipeline.fit(self.X_train_, self.y_train_)
        
        self._store_metrics(self.y_test_, self.pipeline_.predict(self.X_test_))

        return self

    def print_metrics(self):
        """
        Imprime las métricas de evaluación del modelo.
        """
        self._check_fitted()
        if not self.metrics_:
            print("No hay métricas disponibles. Ajuste el modelo primero.")
            return

        print("Métricas de prueba:")
        for metric, value in self.metrics_.items():
            print(f"{metric}: {value:.4f}")

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Importancia de variables (gain), ordenada de mayor a menor.
        """
        self._check_fitted()
        rf = self.pipeline_.named_steps["model"]
        return (
            pd.DataFrame({"Feature": self.feature_names_,
                        "Importancia": rf.feature_importances_})
            .sort_values("Importancia", ascending=False)
            .reset_index(drop=True)
        )