from __future__ import annotations
from .utils import set_seed, train_test_split_df
from .feature_engineering import TABULAR_COLS
from .models.gnn_xgb import train_gnn_xgb
from .models.nlp_transformer import train_nlp
from .models.autoencoder import train_autoencoder
from .models.biometrics import train_biometrics
from .models.ensemble import train_ensemble




class Trainer:
	def __init__(self, artifacts_dir: str = CFG.artifacts_dir):
		self.artifacts_dir = artifacts_dir
		os.makedirs(self.artifacts_dir, exist_ok=True)

	def fit(self, tx: pd.DataFrame, comms: pd.DataFrame, bio: pd.DataFrame, edges: pd.DataFrame):
		set_seed()
		# Split by transactions
		train_df, valid_df = train_test_split_df(tx, test_size=0.2, stratify_col="is_fraud")

		# Align comms & bio
		comms_train = comms[comms.tx_id.isin(train_df.tx_id)]
		comms_valid = comms[comms.tx_id.isin(valid_df.tx_id)]
		bio_train = bio[bio.tx_id.isin(train_df.tx_id)]
		bio_valid = bio[bio.tx_id.isin(valid_df.tx_id)]

		# 1. GNN + XGB (graph-aware tabular)
		gnn_art = train_gnn_xgb(train_df, edges)
		gnn_valid = train_gnn_xgb(valid_df, edges) # small trick: fit on valid too for SHAP parity (demo)
		joblib.dump(gnn_art, os.path.join(self.artifacts_dir, "gnn_xgb.joblib"))
		joblib.dump(gnn_valid, os.path.join(self.artifacts_dir, "gnn_xgb_valid.joblib"))

		# 2. NLP transformer / TF‑IDF
		nlp_art = train_nlp(comms_train, train_df.sort_values("tx_id")["is_fraud"]) # aligned by tx order
		joblib.dump(nlp_art, os.path.join(self.artifacts_dir, "nlp_artifacts.joblib"))

		# 3. Autoencoder anomaly
		ae_art = train_autoencoder(train_df)
		joblib.dump(ae_art, os.path.join(self.artifacts_dir, "autoencoder.joblib"))

		# 4. Biometrics
		bio_art = train_biometrics(bio_train, train_df.sort_values("tx_id")["is_fraud"]) # aligned
		joblib.dump(bio_art, os.path.join(self.artifacts_dir, "biometrics.joblib"))

		# Build expert scores on validation for stacking
		from .models.gnn_xgb import predict_gnn_xgb
		from .models.nlp_transformer import predict_nlp
		from .models.autoencoder import predict_autoencoder
		from .models.biometrics import predict_biometrics

		valid_df_sorted = valid_df.sort_values("tx_id").reset_index(drop=True)
		comms_valid_sorted = comms_valid.sort_values("tx_id").reset_index(drop=True)
		bio_valid_sorted = bio_valid.sort_values("tx_id").reset_index(drop=True)

		s1 = predict_gnn_xgb(gnn_art, valid_df_sorted)
		s2 = predict_nlp(nlp_art, comms_valid_sorted)
		s3 = predict_autoencoder(ae_art, valid_df_sorted)
		s4 = predict_biometrics(bio_art, bio_valid_sorted)

		S_valid = np.vstack([s1, s2, s3, s4]).T
		y_valid = valid_df_sorted["is_fraud"].values

		ens_art = train_ensemble(S_valid, y_valid, backend=CFG.ensemble_type)
		joblib.dump(ens_art, os.path.join(self.artifacts_dir, "ensemble.joblib"))

		# Save feature names for XAI
		joblib.dump(TABULAR_COLS + [f"emb_{i}" for i in range(CFG.graph_embedding_dim)], os.path.join(self.artifacts_dir, "gnn_feature_names.joblib"))

		return {
			"valid_size": len(valid_df_sorted),
			"fraud_rate_valid": float(y_valid.mean()),
			"mean_expert_scores": [float(s1.mean()), float(s2.mean()), float(s3.mean()), float(s4.mean())]
		}

