from __future__ import annotations
import pandas as pd
import numpy as np


TABULAR_COLS = [
"amount","merchant_category","tx_hour","country_mismatch","velocity_1h",
"acct_age_days","kyc_score","device_entropy"
]




def build_tabular_matrix(df: pd.DataFrame) -> np.ndarray:
	X = df[TABULAR_COLS].astype(float).values
	return X