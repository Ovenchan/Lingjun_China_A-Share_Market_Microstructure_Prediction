from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np


@dataclass
class BackendArtifacts:
    train_data: Any
    eval_data: Any


class RegressorBackend:
    model_name = "base"
    model_extension = ".bin"

    def prepare_datasets(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        feature_names: Sequence[str],
    ) -> BackendArtifacts:
        raise NotImplementedError

    def train_epoch(
        self,
        params: Dict[str, Any],
        artifacts: BackendArtifacts,
        booster: Optional[Any],
        num_boost_round: int,
    ) -> Any:
        raise NotImplementedError

    def predict(
        self,
        booster: Any,
        x: np.ndarray,
        feature_names: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        raise NotImplementedError

    def save_model(self, booster: Any, path: str) -> None:
        raise NotImplementedError

    def load_model(self, path: str) -> Any:
        raise NotImplementedError


class LightGBMBackend(RegressorBackend):
    model_name = "lightgbm"
    model_extension = ".txt"

    def __init__(self) -> None:
        import lightgbm as lgb

        self.lgb = lgb

    def prepare_datasets(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        feature_names: Sequence[str],
    ) -> BackendArtifacts:
        train_data = self.lgb.Dataset(
            x_train,
            label=y_train,
            feature_name=list(feature_names),
            free_raw_data=False,
        )
        eval_data = self.lgb.Dataset(
            x_val,
            label=np.zeros(x_val.shape[0], dtype=np.float32),
            reference=train_data,
            free_raw_data=False,
        )
        return BackendArtifacts(train_data=train_data, eval_data=eval_data)

    def train_epoch(
        self,
        params: Dict[str, Any],
        artifacts: BackendArtifacts,
        booster: Optional[Any],
        num_boost_round: int,
    ) -> Any:
        return self.lgb.train(
            params=params,
            train_set=artifacts.train_data,
            num_boost_round=num_boost_round,
            init_model=booster,
            keep_training_booster=True,
        )

    def predict(
        self,
        booster: Any,
        x: np.ndarray,
        feature_names: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        return booster.predict(x)

    def save_model(self, booster: Any, path: str) -> None:
        booster.save_model(path)

    def load_model(self, path: str) -> Any:
        return self.lgb.Booster(model_file=path)


class XGBoostBackend(RegressorBackend):
    model_name = "xgboost"
    model_extension = ".json"

    def __init__(self) -> None:
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError("xgboost is not installed. Please install xgboost first.") from exc

        self.xgb = xgb

    def prepare_datasets(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        feature_names: Sequence[str],
    ) -> BackendArtifacts:
        train_data = self.xgb.DMatrix(x_train, label=y_train, feature_names=list(feature_names))
        eval_data = self.xgb.DMatrix(x_val, feature_names=list(feature_names))
        return BackendArtifacts(train_data=train_data, eval_data=eval_data)

    def train_epoch(
        self,
        params: Dict[str, Any],
        artifacts: BackendArtifacts,
        booster: Optional[Any],
        num_boost_round: int,
    ) -> Any:
        return self.xgb.train(
            params=params,
            dtrain=artifacts.train_data,
            num_boost_round=num_boost_round,
            xgb_model=booster,
        )

    def predict(
        self,
        booster: Any,
        x: np.ndarray,
        feature_names: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        dmatrix = self.xgb.DMatrix(x, feature_names=list(feature_names) if feature_names is not None else None)
        return booster.predict(dmatrix)

    def save_model(self, booster: Any, path: str) -> None:
        booster.save_model(path)

    def load_model(self, path: str) -> Any:
        booster = self.xgb.Booster()
        booster.load_model(path)
        return booster


class CatBoostBackend(RegressorBackend):
    model_name = "catboost"
    model_extension = ".cbm"

    def __init__(self) -> None:
        try:
            from catboost import CatBoostRegressor, Pool
        except ImportError as exc:
            raise ImportError("catboost is not installed. Please install catboost first.") from exc

        self.CatBoostRegressor = CatBoostRegressor
        self.Pool = Pool

    def prepare_datasets(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        feature_names: Sequence[str],
    ) -> BackendArtifacts:
        train_data = self.Pool(x_train, y_train, feature_names=list(feature_names))
        eval_data = self.Pool(x_val, feature_names=list(feature_names))
        return BackendArtifacts(train_data=train_data, eval_data=eval_data)

    def train_epoch(
        self,
        params: Dict[str, Any],
        artifacts: BackendArtifacts,
        booster: Optional[Any],
        num_boost_round: int,
    ) -> Any:
        fit_params = dict(params)
        fit_params["iterations"] = num_boost_round
        model = self.CatBoostRegressor(**fit_params)
        model.fit(artifacts.train_data, init_model=booster, verbose=False)
        return model

    def predict(
        self,
        booster: Any,
        x: np.ndarray,
        feature_names: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        if feature_names is None:
            return booster.predict(x)
        pool = self.Pool(x, feature_names=list(feature_names))
        return booster.predict(pool)

    def save_model(self, booster: Any, path: str) -> None:
        booster.save_model(path)

    def load_model(self, path: str) -> Any:
        model = self.CatBoostRegressor()
        model.load_model(path)
        return model


def create_backend(model_type: str) -> RegressorBackend:
    normalized = model_type.lower()
    if normalized in {"lightgbm", "lgbm", "lgb"}:
        return LightGBMBackend()
    if normalized in {"xgboost", "xgb"}:
        return XGBoostBackend()
    if normalized in {"catboost", "cat"}:
        return CatBoostBackend()
    raise ValueError(f"Unsupported model_type: {model_type}")
