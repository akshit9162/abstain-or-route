"""
Evaluate every finished run from its saved logits. No model is loaded here.

    python -m src.evaluate

For each run and test locale:
  accuracy      with a bootstrap 95% CI over test utterances
  ECE           15 equal-width bins, before and after temperature scaling
  abstention    accuracy on the most-confident X% of requests (the rest go to a human)

Temperature is one scalar per run, fitted on the dev split of the *training* locales only
(English dev for English-only runs), so zero-shot locales never influence it.

MASSIVE test sets are parallel: utterance k is the same request in every locale and in
every run. That makes run-vs-run gaps paired, so they get a paired bootstrap CI instead
of needing extra seeds.

Writes results/summary.json and prints the tables used in the README.
"""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax
from .data import LOCALES

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
B = 1000                       # bootstrap resamples
COVERAGES = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
rng = np.random.default_rng(0)


def load(run):
    z = np.load(RES / run / "logits.npz")
    meta = json.load(open(RES / run / "meta.json"))
    dev = {k.split("__")[1]: (z[k], z[k[:-1] + "y"]) for k in z.files if k.startswith("dev__") and k.endswith("__x")}
    test = {loc: (z[f"test__{loc}__x"], z[f"test__{loc}__y"]) for loc in LOCALES}
    return meta, dev, test


def fit_temperature(x, y):
    """Scalar T minimising dev NLL, searched in log space."""
    nll = lambda logt: -log_softmax(x / np.exp(logt), axis=1)[np.arange(len(y)), y].mean()
    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))


def ece(conf, correct, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def boot_ci(correct):
    n = len(correct)
    means = correct[rng.integers(0, n, (B, n))].mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def selective(conf, correct):
    """Accuracy on the top-c most confident fraction, for each coverage c."""
    order = np.argsort(-conf, kind="stable")
    c = correct[order]
    return {f"{cov:.1f}": float(c[:max(1, int(round(cov * len(c))))].mean()) for cov in COVERAGES}


def evaluate(run):
    meta, dev, test = load(run)
    dx = np.concatenate([v[0] for v in dev.values()])
    dy = np.concatenate([v[1] for v in dev.values()])
    T = fit_temperature(dx, dy)
    out = {"method": meta["method"], "train": meta["train"], "r": meta.get("r"),
           "trainable": meta["trainable"], "total": meta["total"], "T": T, "locales": {}}
    for loc, (x, y) in test.items():
        p0, p1 = softmax(x, axis=1), softmax(x / T, axis=1)
        correct = (x.argmax(1) == y).astype(float)
        out["locales"][loc] = {
            "acc": float(correct.mean()), "ci": boot_ci(correct),
            "ece_raw": ece(p0.max(1), correct), "ece_cal": ece(p1.max(1), correct),
            "selective": selective(p1.max(1), correct),
        }
    L = out["locales"].values()
    out["mean_acc"] = float(np.mean([v["acc"] for v in L]))
    out["mean_ece_raw"] = float(np.mean([v["ece_raw"] for v in L]))
    out["mean_ece_cal"] = float(np.mean([v["ece_cal"] for v in L]))
    return out, test


def paired_gap(test_a, test_b):
    """Mean accuracy of a minus b over all locales, with a paired bootstrap CI over utterances."""
    ca = np.stack([(test_a[l][0].argmax(1) == test_a[l][1]) for l in LOCALES]).astype(float)
    cb = np.stack([(test_b[l][0].argmax(1) == test_b[l][1]) for l in LOCALES]).astype(float)
    d = (ca - cb).mean(0)                      # per-utterance gap, averaged over locales
    n = len(d)
    means = d[rng.integers(0, n, (B, n))].mean(1)
    return {"gap": float(d.mean()), "ci": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]}


def main():
    runs = sorted(p.name for p in RES.iterdir() if (p / "meta.json").exists())
    summary, tests = {"runs": {}, "gaps": {}}, {}
    for run in runs:
        summary["runs"][run], tests[run] = evaluate(run)

    for a, b in [("lora8_en_s0", "full_en_s0"), ("lora8_multi_s0", "full_multi_s0"),
                 ("lora8_en_s0", "probe_en_s0"), ("lora8_multi_s0", "lora8_en_s0"),
                 ("full_multi_s0", "full_en_s0")]:
        if a in tests and b in tests:
            summary["gaps"][f"{a} - {b}"] = paired_gap(tests[a], tests[b])
    json.dump(summary, open(RES / "summary.json", "w"), indent=2)

    short = [l[:2] for l in LOCALES]
    print("Test accuracy (%)")
    print(f"{'run':16s} {'train%':>7s} " + " ".join(f"{s:>5s}" for s in short) + "   mean")
    for run, r in summary["runs"].items():
        accs = [100 * r["locales"][l]["acc"] for l in LOCALES]
        print(f"{run:16s} {100 * r['trainable'] / r['total']:6.2f}% " + " ".join(f"{a:5.1f}" for a in accs)
              + f"  {np.mean(accs):5.1f}")
    print("\nCalibration: mean ECE over 10 locales, raw -> temperature-scaled")
    for run, r in summary["runs"].items():
        worst = max(LOCALES, key=lambda l: r["locales"][l]["ece_cal"])
        print(f"{run:16s} T={r['T']:.2f}  {r['mean_ece_raw']:.3f} -> {r['mean_ece_cal']:.3f}"
              f"   worst after scaling: {worst[:2]} {r['locales'][worst]['ece_cal']:.3f}")
    print("\nPaired gaps in mean accuracy (points, 95% CI)")
    for k, g in summary["gaps"].items():
        print(f"{k:34s} {100 * g['gap']:+5.1f}  [{100 * g['ci'][0]:+.1f}, {100 * g['ci'][1]:+.1f}]")
    print("\nAbstention: accuracy (%) when the least-confident share goes to a human")
    for run, r in summary["runs"].items():
        print(run)
        for l in LOCALES:
            s = r["locales"][l]["selective"]
            print(f"  {l[:2]} " + "  ".join(f"{100 * (1 - float(c)):2.0f}% defer {100 * v:5.1f}" for c, v in s.items()))


if __name__ == "__main__":
    main()
