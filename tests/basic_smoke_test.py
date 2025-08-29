import pandas as pd
from fraudshield.inference import FraudShield


# Minimal smoke test after training


def test_inference_smoke():
	mdl = FraudShield("artifacts/")
	tx = pd.DataFrame([{ "tx_id": 1, "user_id": 5, "amount": 120.0, "merchant_category": 3, "tx_hour": 2,
	"country_mismatch": 1, "velocity_1h": 3, "acct_age_days": 100, "kyc_score": 0.7, "device_entropy": 0.4 }])
	comms = pd.DataFrame([{ "tx_id": 1, "subject": "URGENT: verify", "body": "click this link" }])
	bio = pd.DataFrame([{ "tx_id": 1, "typing_speed": 310, "mouse_jitter": 0.7, "voice_confidence": 0.6 }])
	out = mdl.score_batch(tx, comms, bio)
	assert "risk_overall" in out.columns
	assert out.shape[0] == 1
	print(out)