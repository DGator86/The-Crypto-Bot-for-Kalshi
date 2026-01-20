from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

@dataclass
class XGBArtifact:
    scaler: StandardScaler
    model: XGBClassifier
    feature_names: list[str]

def fit_xgb_classifier(X_train: np.ndarray, y_train: np.ndarray, feature_names: list[str]) -> XGBArtifact:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    
    model = XGBClassifier(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_weight=5,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=0,
    )
    model.fit(Xs, y_train)
    
    return XGBArtifact(scaler=scaler, model=model, feature_names=feature_names)

def predict_proba(artifact: XGBArtifact, X: np.ndarray) -> np.ndarray:
    Xs = artifact.scaler.transform(X)
    return artifact.model.predict_proba(Xs)[:, 1]