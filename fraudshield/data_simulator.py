"""Generate synthetic multi-modal fraud data for demo/training."""

def generate_synthetic_data(n_users, n_tx, fraud_rate, rng, tx, users, nx, SimData, pd, np):
	velocity_1h = rng.integers(0, 8, size=n_tx)
	# Inject coordinated fraud rings (clusters of user_ids)
	ring_centers = rng.choice(n_users, size=10, replace=False)
	ring_members = set()
	for c in ring_centers:
		members = rng.choice(n_users, size=rng.integers(15, 60), replace=False)
		ring_members.update(members)
	ring_members = np.array(list(ring_members))

	# Label fraud
	y = rng.random(n_tx) < fraud_rate
	# increase fraud odds for ring members, high amount, odd hours, country mismatch
	y |= np.isin(tx["user_id"].values, ring_members) & (rng.random(n_tx) < 0.20)
	y |= (tx["amount"].values > np.percentile(tx["amount"], 95)) & (rng.random(n_tx) < 0.15)
	y |= (tx["tx_hour"].isin([0,1,2,3]).values) & (rng.random(n_tx) < 0.10)
	y |= (tx["country_mismatch"] == 1).values & (rng.random(n_tx) < 0.12)
	tx["is_fraud"] = y.astype(int)

	# Communications (NLP)
	comms = pd.DataFrame({
		"tx_id": tx["tx_id"],
		"subject": np.where(tx["is_fraud"].values == 1,
			"URGENT: Action required to verify account",
			"Payment confirmation receipt"),
		"body": np.where(tx["is_fraud"].values == 1,
			"Your account will be closed. Verify identity here: http://phishy.link",
			"Thank you for your payment. No action needed."),
	})

	# Biometrics (toy)
	biometrics = pd.DataFrame({
		"tx_id": tx["tx_id"],
		"typing_speed": rng.normal(250, 30, size=n_tx),
		"mouse_jitter": rng.normal(0.5, 0.1, size=n_tx),
		"voice_confidence": rng.normal(0.85, 0.05, size=n_tx),
	})
	# Slight drift for frauds
	biometrics.loc[tx["is_fraud"] == 1, "typing_speed"] += rng.normal(40, 20, size=(tx["is_fraud"] == 1).sum())
	biometrics.loc[tx["is_fraud"] == 1, "mouse_jitter"] += rng.normal(0.15, 0.05, size=(tx["is_fraud"] == 1).sum())
	biometrics.loc[tx["is_fraud"] == 1, "voice_confidence"] -= rng.normal(0.15, 0.05, size=(tx["is_fraud"] == 1).sum())

	# Graph edges: device->user, user->merchant, user->user transfers
	G = nx.Graph()
	for u in range(n_users):
		G.add_node(f"user_{u}")
	for m in range(300):
		G.add_node(f"merchant_{m}")
	for d in range(800):
		G.add_node(f"device_{d}")

	# connect users to devices and merchants
	for u in range(n_users):
		G.add_edge(f"user_{u}", f"device_{np.random.randint(0,800)}")
		for _ in range(np.random.randint(1,4)):
			G.add_edge(f"user_{u}", f"merchant_{np.random.randint(0,300)}")

	# connect ring members more densely (simulating collusion)
	ring_users = [f"user_{int(u)}" for u in ring_members]
	for i in range(0, len(ring_users), 2):
		if i+1 < len(ring_users):
			G.add_edge(ring_users[i], ring_users[i+1])

	edges = pd.DataFrame([(u, v) for u, v in G.edges()], columns=["src", "dst"])

	# Merge user features into tx
	tx = tx.merge(users, on="user_id", how="left")

	return SimData(tx=tx, comms=comms, biometrics=biometrics, graph_edges=edges)

