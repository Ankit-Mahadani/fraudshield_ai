
# 🔒 FraudShield AI

FraudShield AI is an **open-source fraud detection framework** that combines **Graph Neural Networks, Transformers, Autoencoders, and Ensemble Models** to detect fraudulent transactions, communications, and job postings.

It supports **real datasets** (Kaggle, PaySim, Fake Job Postings) and **synthetic data generation** to simulate fraud scenarios.

---

## 📂 Project Structure

```
fraudshield_ai/
├── README.md
├── requirements.txt
├── fraudshield/
│   ├── config.py
│   ├── utils.py
│   ├── data_simulator.py
│   ├── feature_engineering.py
│   ├── explainability.py
│   ├── training.py
│   ├── inference.py
│   ├── api.py
│   └── models/
│       ├── gnn_xgb.py
│       ├── nlp_transformer.py
│       ├── autoencoder.py
│       ├── biometrics.py
│       └── ensemble.py
├── scripts/
│   ├── generate_synthetic_data.py
│   ├── preprocess_datasets.py
│   ├── run_training.py
│   └── run_api.py
├── data/
│   ├── raw/
│   │   ├── creditcard.csv
│   │   ├── paysim.csv
│   │   └── job_postings.csv
│   └── processed/
└── tests/
    └── basic_smoke_test.py
```

---

## ⚙️ Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/yourusername/fraudshield_ai.git
cd fraudshield_ai
pip install -r requirements.txt
```

---

## 📊 Datasets

### Real Datasets (download & place into `data/raw/`)

* [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) → `creditcard.csv`
* [PaySim Mobile Money Simulation](https://www.kaggle.com/ealaxi/paysim1) → `paysim.csv`
* [Fake Job Postings](https://www.kaggle.com/shivamb/real-or-fake-fake-jobposting-prediction) → `job_postings.csv`

### Synthetic Datasets

Generated automatically using the simulator:

```bash
python scripts/generate_synthetic_data.py
```

Creates:

```
data/processed/
├── transactions.csv
├── comms.csv
├── biometrics.csv
└── graph_edges.csv
```

---

## 🚀 Usage

### 1️⃣ Generate Synthetic Data

```bash
python scripts/generate_synthetic_data.py
```

### 2️⃣ Preprocess Raw Datasets

```bash
python scripts/preprocess_datasets.py
```

### 3️⃣ Train Ensemble Models

```bash
python scripts/run_training.py
```

### 4️⃣ Launch FastAPI Inference Server

```bash
uvicorn scripts.run_api:app --reload
```

---

## ✅ Example Workflow

```bash
# Generate synthetic data
python scripts/generate_synthetic_data.py

# Preprocess raw Kaggle datasets
python scripts/preprocess_datasets.py

# Train fraud detection ensemble
python scripts/run_training.py

# Start API for inference
python scripts/run_api.py
```

---

## 🧪 Tests

Run smoke tests:

```bash
pytest tests/
```

---

## 📖 Roadmap

* [ ] Add streaming fraud detection with Kafka
* [ ] Integrate anomaly detection with PyOD
* [ ] Expand explainability with SHAP + Captum
* [ ] Deploy Docker container for API

---

## 🤝 Contributing

Pull requests are welcome! Please open an issue first to discuss what you’d like to change.

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 🌟 Citation

If you use this repo in research or projects, please cite:

```
FraudShield AI: Modular Framework for Fraud Detection (2025)
```

---
