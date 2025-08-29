import argparse
from pathlib import Path
from fraudshield.data_simulator import make_synthetic_data


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic fraud detection data")
    parser.add_argument(
        "--out",
        type=str,
        default="data/processed/",
        help="Output folder for synthetic datasets"
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)

    # Generate synthetic data
    sim = make_synthetic_data()

    # Save generated datasets
    sim.tx.to_csv(out_path / "transactions.csv", index=False)
    sim.comms.to_csv(out_path / "comms.csv", index=False)
    sim.biometrics.to_csv(out_path / "biometrics.csv", index=False)
    sim.graph_edges.to_csv(out_path / "graph_edges.csv", index=False)

    print(f"✅ Synthetic data saved to {out_path.resolve()}")


if __name__ == "__main__":
    main()
