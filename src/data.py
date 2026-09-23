"""MASSIVE 1.1 loading: 60 intents, parallel utterances across locales."""
import json
import random
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "1.1" / "data"

# Ten locales spanning scripts and language families: Latin (en, de, fr, es, sw),
# Devanagari (hi), Arabic (ar), CJK (ja, zh), Tamil (ta).
LOCALES = ["en-US", "de-DE", "fr-FR", "es-ES", "hi-IN", "ar-SA", "ja-JP", "zh-CN", "sw-KE", "ta-IN"]


def read(locale, partition):
    rows = []
    with open(DATA / f"{locale}.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["partition"] == partition:
                rows.append((r["id"], r["utt"], r["intent"]))
    return rows


def intents():
    """Label set fixed from the English training split, sorted for stable ids."""
    return sorted({i for _, _, i in read("en-US", "train")})


def train_set(mode, seed=0):
    """
    en    : all English training utterances.
    multi : the same number of utterances, split evenly across LOCALES.

    MASSIVE is parallel (the same id is one utterance translated into every locale),
    so 'multi' draws a disjoint set of ids per locale. Otherwise the model would see
    one utterance in ten languages, which is a different experiment.
    """
    en = read("en-US", "train")
    if mode == "en":
        return [(u, i) for _, u, i in en]
    ids = [r[0] for r in en]
    random.Random(seed).shuffle(ids)
    per = len(ids) // len(LOCALES)
    out = []
    for k, loc in enumerate(LOCALES):
        take = set(ids[k * per:(k + 1) * per])
        out += [(u, i) for rid, u, i in read(loc, "train") if rid in take]
    return out


def eval_set(partition, locales=LOCALES):
    return {loc: [(u, i) for _, u, i in read(loc, partition)] for loc in locales}
