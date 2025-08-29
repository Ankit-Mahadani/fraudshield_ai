import argparse
import uvicorn


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("--host", type=str, default="127.0.0.1")
	ap.add_argument("--port", type=int, default=8000)
	ap.add_argument("--model_dir", type=str, default="artifacts/")
	args = ap.parse_args()

	# The FastAPI app loads artifacts from artifacts/
	uvicorn.run("fraudshield.api:app", host=args.host, port=args.port, reload=False)