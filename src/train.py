"""
Train an intent classifier on MASSIVE and dump logits for evaluation.

    python -m src.train --method lora --r 8 --train en --seed 0

method  probe : encoder frozen, only the linear head trains
        lora  : LoRA on attention query/value + head
        full  : every transformer layer trains (word-embedding table frozen, see below)
train   en    : English only (the other 9 locales are zero-shot at test time)
        multi : same number of examples, split evenly over 10 locales

Model selection (best epoch) and later temperature fitting use only the dev split of
the *training* locales, so the zero-shot locales are never looked at before test.
"""
import argparse, json, time, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from . import data
from .lora import inject, count

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "mMiniLM"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class Classifier(nn.Module):
    def __init__(self, n_labels):
        super().__init__()
        self.enc = AutoModel.from_pretrained(MODEL)
        self.head = nn.Linear(self.enc.config.hidden_size, n_labels)

    def forward(self, ids, mask):
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).to(h.dtype)
        return self.head((h * m).sum(1) / m.sum(1).clamp_min(1e-6))    # mean pooling, as the encoder was trained


def batches(tok, rows, label_id, bs, shuffle, rng=None):
    idx = list(range(len(rows)))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        chunk = [rows[j] for j in idx[i:i + bs]]
        enc = tok([u for u, _ in chunk], padding=True, truncation=True, max_length=48, return_tensors="pt")
        y = torch.tensor([label_id[l] for _, l in chunk])
        yield enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE), y.to(DEVICE)


@torch.no_grad()
def logits_for(model, tok, rows, label_id):
    model.eval()
    out, ys = [], []
    for ids, mask, y in batches(tok, rows, label_id, 256, False):
        out.append(model(ids, mask).float().cpu()); ys.append(y.cpu())
    return torch.cat(out).numpy(), torch.cat(ys).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["probe", "lora", "full"], required=True)
    ap.add_argument("--train", choices=["en", "multi"], default="en")
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bs", type=int, default=64)
    a = ap.parse_args()

    name = f"{a.method}{a.r if a.method == 'lora' else ''}_{a.train}_s{a.seed}"
    out = ROOT / "results" / name
    out.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    rng = random.Random(a.seed)

    labels = data.intents()
    label_id = {l: k for k, l in enumerate(labels)}
    train_rows = data.train_set(a.train, seed=a.seed)
    dev_locales = ["en-US"] if a.train == "en" else data.LOCALES
    dev = data.eval_set("dev", dev_locales)
    dev_rows = [r for loc in dev_locales for r in dev[loc]]

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = Classifier(len(labels))
    if a.method == "probe":
        for p in model.enc.parameters():
            p.requires_grad_(False)
        groups = [{"params": model.head.parameters(), "lr": 1e-3}]
    elif a.method == "lora":
        n_wrapped = inject(model.enc, r=a.r, alpha=2 * a.r)
        lora_params = [p for n, p in model.enc.named_parameters() if p.requires_grad]
        groups = [{"params": lora_params, "lr": 1e-3}, {"params": model.head.parameters(), "lr": 1e-3}]
    else:
        # The 250k-token embedding table is 96M of the encoder's 118M parameters. Its Adam state
        # alone does not fit next to everything else in 8 GB, and a batch touches only a few
        # hundred of its rows, so it stays frozen; every transformer layer trains.
        model.enc.embeddings.word_embeddings.weight.requires_grad_(False)
        enc_params = [p for p in model.enc.parameters() if p.requires_grad]
        groups = [{"params": enc_params, "lr": 3e-5}, {"params": model.head.parameters(), "lr": 1e-3}]
    model.to(DEVICE)
    trainable, total = count(model)

    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    steps = a.epochs * ((len(train_rows) + a.bs - 1) // a.bs)
    warm = int(0.06 * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, warm)) * max(0.0, (steps - s) / max(1, steps - warm)))
    lossf = nn.CrossEntropyLoss()

    print(f"{name}: {len(train_rows)} train, trainable {trainable:,} / {total:,} "
          f"({100 * trainable / total:.2f}%), device {DEVICE}", flush=True)
    best, best_state, curve, t0 = -1.0, None, [], time.time()
    for ep in range(a.epochs):
        model.train()
        if a.method == "probe":
            model.enc.eval()      # frozen feature extractor: no dropout, fixed features
        for ids, mask, y in batches(tok, train_rows, label_id, a.bs, True, rng):
            loss = lossf(model(ids, mask), y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        lg, y = logits_for(model, tok, dev_rows, label_id)
        acc = float((lg.argmax(1) == y).mean())
        curve.append(acc)
        print(f"  epoch {ep + 1}: dev acc {acc:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        if acc > best:
            best = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    dev_lg = {loc: logits_for(model, tok, dev[loc], label_id) for loc in dev_locales}
    test = data.eval_set("test")
    test_lg = {loc: logits_for(model, tok, test[loc], label_id) for loc in data.LOCALES}
    np.savez_compressed(out / "logits.npz",
                        **{f"dev__{l}__x": v[0] for l, v in dev_lg.items()},
                        **{f"dev__{l}__y": v[1] for l, v in dev_lg.items()},
                        **{f"test__{l}__x": v[0] for l, v in test_lg.items()},
                        **{f"test__{l}__y": v[1] for l, v in test_lg.items()})
    meta = dict(vars(a), name=name, trainable=trainable, total=total, n_train=len(train_rows),
                dev_curve=curve, best_dev=best, seconds=round(time.time() - t0, 1),
                test_acc={l: float((v[0].argmax(1) == v[1]).mean()) for l, v in test_lg.items()})
    json.dump(meta, open(out / "meta.json", "w"), indent=2)
    print(f"  test acc: " + "  ".join(f"{l[:2]} {v:.3f}" for l, v in meta["test_acc"].items()), flush=True)


if __name__ == "__main__":
    main()
