from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

@dataclass
class XGBRegArtifact:
    scaler: StandardScaler
    model: XGBRegressor
    feature_names: list[str]

def fit_xgb_regressor(X_train: np.ndarray, y_train: np.ndarray, feature_names: list[str]) -> XGBRegArtifact:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    
    model = XGBRegressor(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_weight=5,
        objective="reg:squarederror",
        n_jobs=0,
    )
    model.fit(Xs, y_train)
    
    return XGBRegArtifact(scaler=scaler, model=model, feature_names=feature_names)

def predict_reg(artifact: XGBRegArtifact, X: np.ndarray) -> np.ndarray:
    Xs = artifact.scaler.transform(X)
    return artifact.model.predict(Xs)