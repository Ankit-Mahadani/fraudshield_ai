from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import numpy as np

HF_OK = True
try:
	from transformers import AutoTokenizer, AutoModelForSequenceClassification
	import torch
except Exception:
	HF_OK = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

@dataclass
class NLPArtifacts:
	backend: str # "transformer" or "tfidf"
	tokenizer: object | None
	model: object
	vectorizer: object | None

def train_nlp(comms: pd.DataFrame, labels: pd.Series) -> NLPArtifacts:
	texts = (comms["subject"].fillna("") + " \n " + comms["body"].fillna("")).tolist()

	if CFG.use_transformers and HF_OK:
		tokenizer = AutoTokenizer.from_pretrained(CFG.nlp_model_name)
		model = AutoModelForSequenceClassification.from_pretrained(CFG.nlp_model_name, num_labels=2)
		# Quick linear-probe style fine-tune using simple batching (demo)
		enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
		y = torch.tensor(labels.values)
		optim = torch.optim.AdamW(model.parameters(), lr=2e-5)
		model.train()
		for _ in range(2): # few epochs for demo
			optim.zero_grad()
			out = model(**enc, labels=y)
			out.loss.backward()
			optim.step()
		model.eval()
		return NLPArtifacts("transformer", tokenizer, model, None)

	# Fallback: TF‑IDF + LogisticRegression
	vec = TfidfVectorizer(min_df=2, ngram_range=(1,2))
	X = vec.fit_transform(texts)
	clf = LogisticRegression(max_iter=200)
	clf.fit(X, labels.values)
	return NLPArtifacts("tfidf", None, clf, vec)




def predict_nlp(art: NLPArtifacts, comms_batch: pd.DataFrame) -> np.ndarray:
	texts = (comms_batch["subject"].fillna("") + " \n " + comms_batch["body"].fillna("")).tolist()
	if art.backend == "transformer":
		import torch
		enc = art.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
		with torch.no_grad():
			logits = art.model(**enc).logits
			probs = torch.softmax(logits, dim=-1)[:,1].cpu().numpy()
		return probs
	else:
		X = art.vectorizer.transform(texts)
		return art.model.predict_proba(X)[:,1]

