import re
from fractions import Fraction
from pathlib import Path
import numpy as np
from common import parse_answer

WORDS = "therefore however wait alternatively approximately remaining total each half percent incorrect mistake assume cannot not so boxed frac times div text".split()


def features(bank, arrays):
    rows = []
    for q in bank:
        texts = [c["text"] for c in q["candidates"]]
        answers = [parse_answer(t) for t in texts]
        lens = np.array([len(t) for t in texts])
        for i, t in enumerate(texts):
            a = answers[i]
            v = float(Fraction(a)) if a is not None else 0.0
            low = t.lower()
            rows.append(
                [
                    a is not None,
                    sum(a == b for b in answers) if a else 0,
                    len(set(answers)),
                    np.log1p(abs(v)),
                    v < 0,
                    v == int(v),
                    len(t),
                    len(t) / (lens.mean() + 1),
                    len(t.split()),
                    t.count("\n"),
                    t.count("="),
                    t.count("$"),
                    len(re.findall(r"\d+", t)),
                    len(q["question"]),
                    *[low.count(w) for w in WORDS],
                ]
            )
    f = np.asarray(rows, dtype=np.float32)
    e = arrays["embeddings"]
    return np.concatenate(
        [f, e.reshape(-1, 1024), (e - e.mean(axis=1, keepdims=True)).reshape(-1, 1024)],
        axis=1,
    )


def predict(bank, arrays, checkpoint):
    from catboost import CatBoostClassifier

    model = CatBoostClassifier()
    model.load_model(str(Path(checkpoint) / "model.cbm"))
    x = features(bank, arrays)
    p = model.predict_proba(x, thread_count=4)[:, 1].reshape(-1, 4)
    p[x[:, 0].reshape(-1, 4) == 0] = -1
    return p.argmax(axis=1).tolist()
