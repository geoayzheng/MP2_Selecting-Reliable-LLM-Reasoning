import os
from pathlib import Path
import numpy as np
from common import load_inputs, read_jsonl, parse_answer
from attention_model import inputs

DATA = Path(
    os.environ.get(
        "MP2_DATA_DIR", str(Path(__file__).resolve().parent.parent / "code/data")
    )
)


def load(split):
    b, a = load_inputs(split, data_dir=DATA)
    x = inputs(b, a)
    ans = [[parse_answer(c["text"]) for c in q["candidates"]] for q in b]
    refs = read_jsonl(DATA / f"gsm8k_{split}.jsonl")
    gold = [parse_answer(r["answer"]) for r in refs]
    y = np.array(
        [[int(v is not None and v == g) for v in aa] for aa, g in zip(ans, gold)],
        dtype=np.float32,
    )
    valid = np.array([[v is not None for v in aa] for aa in ans])
    return b, a, x, y, valid, ans
