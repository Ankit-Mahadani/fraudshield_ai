from __future__ import annotations
import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict
from .models.gnn_xgb import predict_gnn_xgb
from .models.nlp_transformer import predict_nlp
from .models.autoencoder import predict_autoencoder
from .models.biometrics import predict_biometrics
from .models.ensemble import predict_ensemble
from .explainability import shap_top_features_xgb




class FraudShield:
	def __init__(self, artifacts_dir: str):
		self.dir = artifacts_dir
		self.gnn_art = joblib.load(os.path.join(self.dir, "gnn_xgb.joblib"))
		self.gnn_feature_names = joblib.load(os.path.join(self.dir, "gnn_feature_names.joblib"))
		self.nlp_art = joblib.load(os.path.join(self.dir, "nlp_artifacts.joblib"))
		self.ae_art = joblib.load(os.path.join(self.dir, "autoencoder.joblib"))
		self.bio_art = joblib.load(os.path.join(self.dir, "biometrics.joblib"))
		self.ens_art = joblib.load(os.path.join(self.dir, "ensemble.joblib"))

	def score_batch(self, tx: pd.DataFrame, comms: pd.DataFrame, bio: pd.DataFrame) -> pd.DataFrame:
		s1 = predict_gnn_xgb(self.gnn_art, tx)
		s2 = predict_nlp(self.nlp_art, comms)
		s3 = predict_autoencoder(self.ae_art, tx)
		s4 = predict_biometrics(self.bio_art, bio)
		S = np.vstack([s1, s2, s3, s4]).T
		p = predict_ensemble(self.ens_art, S)

		# Explainability (top-3 reasons from GNN+XGB)
		import numpy as np
		reasons = []
		for i in range(len(tx)):
			# Rebuild feature vector for this row
			emb = self.gnn_art.user_emb.get(int(tx.iloc[i].user_id), np.zeros(self.gnn_art.emb_dim))
			x_tab = tx.iloc[i][self.gnn_feature_names[:len(self.gnn_feature_names)-self.gnn_art.emb_dim]].astype(float).values
			X_row = np.hstack([x_tab, emb])
			r = shap_top_features_xgb(self.gnn_art.xgb_model, X_row, self.gnn_feature_names)
			reasons.append(r)

		return pd.DataFrame({
			"tx_id": tx["tx_id"].values,
			"risk_gnn": s1,
			"risk_nlp": s2,
			"risk_ae": s3,
			"risk_bio": s4,
			"risk_overall": p,
			"reasons": reasons,
			"decision": np.where(p>0.85, "BLOCK+ESCALATE", np.where(p>0.65, "CHALLENGE_2FA", "APPROVE"))
		})

