from __future__ import annotations


def _compute_user_embeddings(edges, users, emb_dim):
	# Build graph and mapping
	import networkx as nx
	G = nx.from_pandas_edgelist(edges, "user_id_1", "user_id_2")
	mapping = {name: idx for idx, name in enumerate(G.nodes())}
	n_users_graph = len(G.nodes())

	# Create random initial features
	x = np.random.RandomState(CFG.seed).randn(n_users_graph, emb_dim).astype("float32")

	# Build edge index
	edges_idx = np.array([[mapping[u], mapping[v]] for u, v in G.edges()], dtype=np.int64)
	if edges_idx.size == 0:
		edges_idx = np.zeros((0,2), dtype=np.int64)

	import torch
	edge_index = torch.tensor(edges_idx.T if edges_idx.size else np.zeros((2,0),dtype=np.int64))
	x = torch.tensor(x)

	# Simple two-layer GraphSAGE
	conv1 = SAGEConv(emb_dim, emb_dim)
	conv2 = SAGEConv(emb_dim, emb_dim)
	with torch.no_grad():
		x = conv1(x, edge_index).relu()
		x = conv2(x, edge_index)

	# Map back to user_id
	user_emb = {}
	for name, idx in mapping.items():
		uid = int(str(name).split("_")[1])
		user_emb[uid] = x[idx].numpy()

	# Users not in graph
	rng = np.random.default_rng(CFG.seed)
	for uid in users["user_id"].values:
		if int(uid) not in user_emb:
			user_emb[int(uid)] = rng.normal(0, 1, size=emb_dim)

	return user_emb




def train_gnn_xgb(tx: pd.DataFrame, edges: pd.DataFrame) -> GNNXGBArtifacts:
	users = tx[["user_id"]].drop_duplicates().reset_index(drop=True)
	user_emb = _compute_user_embeddings(edges, users, CFG.graph_embedding_dim)

	# Build training matrix: tabular + user embedding
	emb_mat = np.vstack([user_emb[int(u)] for u in tx["user_id"].values])
	X_tab = build_tabular_matrix(tx)
	X = np.hstack([X_tab, emb_mat])
	y = tx["is_fraud"].values

	model = xgb.XGBClassifier(
		n_estimators=200,
		max_depth=5,
		learning_rate=0.1,
		subsample=0.8,
		colsample_bytree=0.8,
		reg_lambda=1.0,
		n_jobs=4,
		random_state=CFG.seed,
	)
	model.fit(X, y)

	return GNNXGBArtifacts(xgb_model=model, user_emb=user_emb, emb_dim=CFG.graph_embedding_dim)




def predict_gnn_xgb(art: GNNXGBArtifacts, tx_batch: pd.DataFrame) -> np.ndarray:
	emb_mat = np.vstack([art.user_emb.get(int(u), np.zeros(art.emb_dim)) for u in tx_batch["user_id"].values])
	X_tab = build_tabular_matrix(tx_batch)
	X = np.hstack([X_tab, emb_mat])
	return art.xgb_model.predict_proba(X)[:,1]

