import os

for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[key] = "4"
import json, time
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import common

common.DATA = Path(__file__).resolve().parent.parent / "code/data"
from train_attention import load
from attention_model import SetVerifier, probabilities, pooled
from catboost import CatBoostRanker
from baseline_submission import features

root = Path(__file__).resolve().parent
out = root / "retrained_checkpoint"
out.mkdir(exist_ok=True)
b, a, x, y, valid, ans = load("train")
vb, va, v, vy, vvalid, vans = load("validation")
columns = np.r_[np.arange(1059), np.arange(2083, 2091)]
x = x[:, :, columns]
v = v[:, :, columns]
sc = StandardScaler().fit(x.reshape(-1, x.shape[-1]))
xx = np.clip((x - sc.mean_) / sc.scale_, -8, 8).astype(np.float32)
vv = np.clip((v - sc.mean_) / sc.scale_, -8, 8).astype(np.float32)
tx = torch.from_numpy(xx)
ty = torch.from_numpy(y)
tv = torch.from_numpy(valid)
vx = torch.from_numpy(vv)
checkpoints = [5]
scores = {e: [] for e in checkpoints}
records = []
start = time.perf_counter()
for seed in [42, 123, 2026]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SetVerifier(x.shape[-1], 128)
    opt = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.08)
    for epoch in range(1, 6):
        model.train()
        for batch in np.array_split(rng.permutation(len(b)), int(np.ceil(len(b) / 64))):
            s = model(tx[batch])
            target = ty[batch]
            mask = tv[batch]
            pairs = target[:, :, None] * (1 - target[:, None, :]) * mask[:, None, :]
            rank = (
                torch.nn.functional.softplus(-(s[:, :, None] - s[:, None, :])) * pairs
            ).sum((1, 2)) / pairs.sum((1, 2)).clamp(min=1)
            mass = torch.logsumexp(s.masked_fill(~mask, -1e4), 1) - torch.logsumexp(
                s.masked_fill(target == 0, -1e4), 1
            )
            loss = (
                0.5 * rank.mean()
                + 0.5 * mass.mean()
                + 0.15 * torch.nn.functional.binary_cross_entropy_with_logits(s, target)
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
            opt.step()
        if epoch in checkpoints:
            model.eval()
            with torch.no_grad():
                p = probabilities(model(vx).numpy(), vvalid)
            scores[epoch].append(p)
            record = {
                "seed": seed,
                "epoch": epoch,
                "validation": float(
                    vy[np.arange(len(vy)), pooled(p, vans, 1).argmax(1)].mean()
                ),
            }
            records.append(record)
            print(record, flush=True)
            torch.save(
                {
                    "dim": x.shape[-1],
                    "width": 128,
                    "columns": torch.tensor(columns),
                    "mean": torch.tensor(sc.mean_.astype(np.float32)),
                    "scale": torch.tensor(sc.scale_.astype(np.float32)),
                    "state": model.state_dict(),
                },
                out / f"attention_{seed}.pt",
            )

from sklearn.linear_model import LogisticRegression

raw = features(b, a)
scaler = StandardScaler().fit(raw)
scaled = scaler.transform(raw).reshape(len(b), 4, -1)
pairs = []
for group, labels in zip(scaled, y):
    for i in range(4):
        for j in range(4):
            if labels[i] > labels[j]:
                pairs.append(group[i] - group[j])
pairs = np.array(pairs)
rank = LogisticRegression(C=0.001, max_iter=2000, fit_intercept=False).fit(
    np.r_[pairs, -pairs], np.r_[np.ones(len(pairs)), np.zeros(len(pairs))]
)
np.savez(
    out / "rank.npz",
    coef=rank.coef_,
    intercept=np.array([0.0]),
    mean=scaler.mean_,
    scale=scaler.scale_,
)
config = json.loads((root / "checkpoint/config.json").read_text())
(out / "config.json").write_text(json.dumps(config, indent=2))
print("Fixed recipe complete; original checkpoint preserved. Output:", out)
