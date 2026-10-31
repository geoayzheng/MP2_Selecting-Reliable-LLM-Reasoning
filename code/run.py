import os, sys, json, time, subprocess, argparse, hashlib
from pathlib import Path

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", type=Path, default=root.parent / "code/data")
parser.add_argument("--split", choices=["test", "validation"], default="test")
args = parser.parse_args()
for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[key] = "4"
manifest = json.loads((root / "MANIFEST.json").read_text())
for name, digest in manifest["sha256"].items():
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
        raise RuntimeError("File hash mismatch: " + name)
out = root / "results"
out.mkdir(exist_ok=True)
pred = out / f"{args.split}-predictions.json"
score = out / f"{args.split}-score.json"
started = time.perf_counter()
subprocess.run(
    [
        sys.executable,
        str(root / "predict.py"),
        "--method",
        str(root / "submission_attention.py"),
        "--checkpoint",
        str(root / "checkpoint"),
        "--data-dir",
        str(args.data_dir),
        "--split",
        args.split,
        "--output",
        str(pred),
    ],
    check=True,
)
subprocess.run(
    [
        sys.executable,
        str(root / "score_predictions.py"),
        "--data-dir",
        str(args.data_dir),
        "--split",
        args.split,
        "--predictions",
        str(pred),
        "--output",
        str(score),
    ],
    check=True,
)
seconds = time.perf_counter() - started
result = json.loads(score.read_text())
result["external_prediction_and_scoring_seconds"] = seconds
(out / f"{args.split}-measurement.json").write_text(json.dumps(result, indent=2))
print("Final accuracy:", result["accuracy"], "%; elapsed:", round(seconds, 3), "s")
