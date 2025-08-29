import argparse
from pathlib import Path
import pandas as pd
from fraudshield.training import Trainer
from fraudshield.config import CFG


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("--data", type=str, default="data/")
	ap.add_argument("--artifacts", type=str, default="artifacts/")
	args = ap.parse_args()

	CFG.ensure_dirs()

	tx = pd.read_csv(Path(args.data)/"transactions.csv")
	comms = pd.read_csv(Path(args.data)/"comms.csv")
	bio = pd.read_csv(Path(args.data)/"biometrics.csv")
	edges = pd.read_csv(Path(args.data)/"graph_edges.csv")

	t = Trainer(artifacts_dir=args.artifacts)
	summary = t.fit(tx, comms, bio, edges)
	print("Training summary:", summary)