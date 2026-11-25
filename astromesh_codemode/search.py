"""Lexical BM25 index over the tool catalog; backs the `search` tool."""

from __future__ import annotations

import math
import re
from collections import Counter

from astromesh_codemode.spec import ToolSpec

MAX_LIMIT = 25
K1 = 1.5
B = 0.75

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_SPLIT = re.compile(r"[^A-Za-z0-9]+")


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _SPLIT.split(_CAMEL.sub(" ", text)) if t]


def _doc_tokens(spec: ToolSpec) -> list[str]:
    props = (spec.input_schema or {}).get("properties", {}) or {}
    out = _tokens(spec.name) + _tokens(spec.description or "")
    for p in props:
        out += _tokens(str(p))
    return out


class Index:
    def __init__(self, specs: list[ToolSpec]):
        self._specs = list(specs)
        self._tf = [Counter(_doc_tokens(s)) for s in self._specs]
        self._len = [sum(tf.values()) for tf in self._tf]
        n = len(self._specs)
        self._avg = (sum(self._len) / n) if n else 0.0
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        self._idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def search(self, query: str, limit: int = 8) -> list[ToolSpec]:
        limit = max(0, min(int(limit), MAX_LIMIT))
        terms = list(dict.fromkeys(_tokens(query)))
        if not terms:
            return self._specs[:limit]
        scored: list[tuple[float, int]] = []
        for i, tf in enumerate(self._tf):
            norm = K1 * (1 - B + B * self._len[i] / self._avg) if self._avg else K1
            score = 0.0
            for t in terms:
                f = tf.get(t, 0)
                if f:
                    score += self._idf[t] * f * (K1 + 1) / (f + norm)
            if score > 0:
                scored.append((score, i))
        scored.sort(key=lambda p: (-p[0], p[1]))
        return [self._specs[i] for _, i in scored[:limit]]
