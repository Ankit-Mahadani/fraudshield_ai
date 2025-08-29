from __future__ import annotations
import numpy as np
from dataclasses import dataclass


from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb


@dataclass
class EnsembleArtifacts:
	backend: str # "logistic" or "xgb"
	model: object




def train_ensemble(expert_scores: np.ndarray, labels: np.ndarray, backend: str = "logistic") -> EnsembleArtifacts:
	# expert_scores shape: [N, num_experts]
	if backend == "xgb":
		mdl = xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9, random_state=42)
		mdl.fit(expert_scores, labels)
		return EnsembleArtifacts("xgb", mdl)

	# default logistic + calibration
	base = LogisticRegression(max_iter=200)
	mdl = CalibratedClassifierCV(base, method="sigmoid", cv=3)
	mdl.fit(expert_scores, labels)
	return EnsembleArtifacts("logistic", mdl)




def predict_ensemble(art: EnsembleArtifacts, expert_scores: np.ndarray) -> np.ndarray:
	return art.model.predict_proba(expert_scores)[:,1]