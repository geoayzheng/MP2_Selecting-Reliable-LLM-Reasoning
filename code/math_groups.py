"""Conservative numeric-equation checks and answer-group representations."""

import ast
import operator
import re
from fractions import Fraction
import numpy as np
from common import parse_answer
from baseline_submission import features

OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
NUM = r"(?:\d+(?:\.\d+)?)"
EXPR = re.compile(
    r"(?<![\w.])("
    + NUM
    + r"(?:\s*[+*/-]\s*"
    + NUM
    + r")+)\s*=\s*([-+]?"
    + NUM
    + r")(?![\w.])"
)


def safe_number(expression):
    def walk(node):
        if isinstance(node, ast.Constant) and type(node.value) in (float, int):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return OPS[type(node.op)](walk(node.left), walk(node.right))
        raise ValueError("Unsupported expression")

    return walk(ast.parse(expression, mode="eval").body)


def arithmetic(text):
    t = (
        text.replace(",", "")
        .replace("\\times", "*")
        .replace("\\cdot", "*")
        .replace("×", "*")
        .replace("÷", "/")
        .replace("\\div", "/")
        .replace("−", "-")
    )
    t = re.sub(r"\\(?:d?frac)\{([-+]?\d+(?:\.\d+)?)\}\{(\d+(?:\.\d+)?)\}", r"\1/\2", t)
    # Keep parentheses and variables as boundaries: do not claim to verify symbolic algebra.
    t = t.replace("$", "").replace("**", "")
    errors = []
    for m in EXPR.finditer(t):
        # Avoid checking a suffix of a parenthesized or symbolic expression.
        left = t[: m.start()].rstrip()
        if left and (left[-1] in "+-*/(" or left[-1].isalnum()):
            continue
        right = t[m.end() :].lstrip()
        if right.startswith(("+", "-", "*", "/", ".")):
            continue
        try:
            a = safe_number(m.group(1))
            b = float(m.group(2))
            errors.append(abs(a - b) / max(1, abs(a), abs(b)))
        except (ValueError, ZeroDivisionError, OverflowError, SyntaxError):
            pass
    e = np.array(errors)
    bad = int((e > 1e-3).sum())
    return [
        len(e),
        bad,
        len(e) - bad,
        bad / max(1, len(e)),
        float(e.max()) if len(e) else 0,
        float(e.mean()) if len(e) else 0,
        t.count("="),
        len(e) / max(1, t.count("=")),
    ]


def grouped(bank, arrays):
    base = features(bank, arrays).reshape(len(bank), 4, -1)
    rows = []
    owners = []
    choices = []
    answers = []
    for k, q in enumerate(bank):
        aa = [parse_answer(c["text"]) for c in q["candidates"]]
        math = np.array([arithmetic(c["text"]) for c in q["candidates"]])
        for a in dict.fromkeys(aa):
            if a is None:
                continue
            ii = [i for i, b in enumerate(aa) if a == b]
            f = base[k, ii]
            m = math[ii]
            # First 131 dimensions are statistical/group features; remaining dimensions are mean embeddings.
            row = np.r_[
                len(ii),
                len(set(aa) - {None}),
                f[:, :35].mean(0),
                f[:, :35].min(0),
                f[:, :35].max(0),
                m.mean(0),
                m.min(0),
                m.max(0),
                f[:, 35:].mean(0),
            ]
            rows.append(row)
            owners.append(k)
            choices.append(ii[0])
            answers.append(a)
    return (
        np.array(rows, dtype=np.float64),
        np.array(owners),
        np.array(choices),
        answers,
    )


def select(scores, owners, choices, n):
    result = np.zeros(n, dtype=int)
    for k in range(n):
        ii = np.flatnonzero(owners == k)
        if len(ii):
            result[k] = choices[ii[np.argmax(scores[ii])]]
    return result.tolist()
