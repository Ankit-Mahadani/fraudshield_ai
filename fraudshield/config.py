from dataclasses import dataclass
from pathlib import Path


@dataclass
class CFG:
	seed: int = 42
	artifacts_dir: str = "artifacts"
	nlp_model_name: str = "distilbert-base-uncased" # will fallback to TF-IDF if unavailable
	use_transformers: bool = True
	use_pyg: bool = True
	max_train_rows: int = 20000 # synthetic demo scale
	graph_embedding_dim: int = 64
	autoencoder_latent: int = 16
	ensemble_type: str = "logistic" # or "xgb"

	@staticmethod
	def ensure_dirs():
		Path(CFG.artifacts_dir).mkdir(parents=True, exist_ok=True)