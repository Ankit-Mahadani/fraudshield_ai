from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel
from .inference import FraudShield
import pandas as pd


app = FastAPI(title="FraudShield AI", version="0.1")
model: FraudShield | None = None


class TxIn(BaseModel):
	tx_id: int
	user_id: int
	amount: float
	merchant_category: int
	tx_hour: int
	country_mismatch: int
	velocity_1h: int
	acct_age_days: int
	kyc_score: float
	device_entropy: float
	subject: str
	body: str
	typing_speed: float
	mouse_jitter: float
	voice_confidence: float


@app.on_event("startup")
async def _load_model():
	global model
	model = FraudShield("artifacts/")


@app.post("/score")
def score(item: TxIn):
	tx = pd.DataFrame([{k: getattr(item, k) for k in [
		"tx_id","user_id","amount","merchant_category","tx_hour","country_mismatch","velocity_1h","acct_age_days","kyc_score","device_entropy"
	]}])
	comms = pd.DataFrame([{k: getattr(item, k) for k in ["tx_id","subject","body"]}])
	bio = pd.DataFrame([{k: getattr(item, k) for k in ["tx_id","typing_speed","mouse_jitter","voice_confidence"]}])
	out = model.score_batch(tx, comms, bio)
	return out.to_dict(orient="records")[0]

