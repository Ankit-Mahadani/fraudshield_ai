import random
import numpy as np
import pandas as pd


from .config import CFG

def set_seed(seed: int = CFG.seed):
	random.seed(seed)
	np.random.seed(seed)
	import torch
	try:
		torch.manual_seed(seed)
	except Exception:
		pass




def train_test_split_df(df: pd.DataFrame, test_size: float = 0.2, stratify_col: str | None = None):
	from sklearn.model_selection import train_test_split
	stratify = df[stratify_col] if stratify_col and stratify_col in df.columns else None
	tr, te = train_test_split(df, test_size=test_size, random_state=CFG.seed, stratify=stratify)
	return tr.reset_index(drop=True), te.reset_index(drop=True)