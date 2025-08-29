from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier


@dataclass
class BioArtifacts:
    model: RandomForestClassifier
    cols: list[str]




def train_biometrics(bio: pd.DataFrame, labels: pd.Series) -> BioArtifacts:
    cols = [c for c in bio.columns if c != "tx_id"]
    X = bio[cols].values
    y = labels.values
    rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    rf.fit(X, y)
    return BioArtifacts(model=rf, cols=cols)




def predict_biometrics(art: BioArtifacts, bio_batch: pd.DataFrame) -> np.ndarray:
    X = bio_batch[art.cols].values
    return art.model.predict_proba(X)[:,1]

