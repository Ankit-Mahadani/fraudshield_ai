from __future__ import annotations
import numpy as np
import shap




def shap_top_features_xgb(xgb_model, X_row: np.ndarray, feature_names: list[str], top_k: int = 3) -> list[str]:
	"""Return short human-readable reasons from SHAP values for an XGBoost model."""
	try:
		explainer = shap.TreeExplainer(xgb_model)
		sv = explainer.shap_values(X_row.reshape(1,-1))
		if isinstance(sv, list):
			sv = sv[1]
		sv = sv.flatten()
		idx = np.argsort(np.abs(sv))[::-1][:top_k]
		reasons = [f"{feature_names[i]} impact: {sv[i]:+.3f}" for i in idx]
		return reasons
	except Exception:
		return ["Explainability unavailable (fallback mode)"]