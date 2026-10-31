"""Three locally trained attention seeds plus a frozen train-only ranker."""

import json
from pathlib import Path
import numpy as np
import torch
from common import parse_answer
from attention_model import SetVerifier, inputs, probabilities, pooled
from baseline_submission import features


def predict(bank, arrays, checkpoint):
    torch.set_num_threads(4)
    root = Path(checkpoint)
    config = json.loads((root / "config.json").read_text())
    x = inputs(bank, arrays)
    answers = [[parse_answer(c["text"]) for c in q["candidates"]] for q in bank]
    valid = np.array([[a is not None for a in aa] for aa in answers])
    ps = []
    for seed in config["seeds"]:
        bundle = torch.load(
            root / f"attention_{seed}.pt", map_location="cpu", weights_only=True
        )
        if bundle.get("kind") == "looped":
            from lecture_model import LoopedVerifier

            model = LoopedVerifier(bundle["dim"], bundle.get("width", 128))
        else:
            model = SetVerifier(bundle["dim"], bundle.get("width", 96))
        model.load_state_dict(bundle["state"])
        model.eval()
        member_x = x[:, :, bundle["columns"].numpy()] if "columns" in bundle else x
        xx = np.clip(
            (member_x - bundle["mean"].numpy()) / bundle["scale"].numpy(), -8, 8
        ).astype(np.float32)
        with torch.no_grad():
            p = probabilities(model(torch.from_numpy(xx)).numpy(), valid)
        ps.append(p)
    p = np.mean(ps, axis=0)
    w = config["attention_weight"]
    if w < 1:
        f = features(bank, arrays)
        if config.get("rank_kind") == "linear":
            with np.load(root / "rank.npz") as rank:
                logits = (
                    ((f - rank["mean"]) / rank["scale"]) @ rank["coef"].T
                    + rank["intercept"]
                ).reshape(-1, 4)
        else:
            from catboost import CatBoostRanker

            rank = CatBoostRanker()
            rank.load_model(str(root / "rank.cbm"))
            logits = rank.predict(f, thread_count=4).reshape(-1, 4)
        r = probabilities(logits, valid)
        p = w * p + (1 - w) * r
    p = pooled(p, answers, config["pooling"])
    p[~valid] = -1
    return p.argmax(1).tolist()
