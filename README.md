# Abstain or Route: multilingual intent routing that knows when to hand off

**[→ Visual walkthrough and results](https://akshit9162.github.io/abstain-or-route/)**

A support system gets requests in many languages: *"my alarm didn't go off"*, *"wecke mich um sieben"*,
*"मेरा अलार्म सेट करो"*. Each has to be routed to the right intent, and a wrong route costs more than a
slow one. So for every request the model either **routes** it or **abstains** and hands it to a human.

This repo measures three things on that problem:

1. **Parameter-efficient fine-tuning.** How close does LoRA get to full fine-tuning? LoRA is written
   from scratch in PyTorch ([`src/lora.py`](src/lora.py)), not imported.
2. **Cross-lingual transfer.** What does training on English alone cost in other languages, compared with a
   multilingual mix of the same size?
3. **Knowing when not to answer.** Is the model's confidence trustworthy in every language, and how much
   does handing the least-confident requests to a human help?

## Setup

| | |
|---|---|
| Data | [MASSIVE 1.1](https://github.com/alexa/massive) (Amazon Science, CC BY 4.0): 60 intents, parallel across languages |
| Languages | 10, chosen to differ in script and family: en, de, fr, es, sw (Latin), hi (Devanagari), ar (Arabic), ja, zh (CJK), ta (Tamil) |
| Encoder | `paraphrase-multilingual-MiniLM-L12-v2` (118M parameters, mean pooling), plus a linear head |
| Training data | **en**: all 11,514 English training utterances. **multi**: 11,510 utterances split evenly over the 10 languages, drawn from disjoint utterance ids (MASSIVE is parallel, so otherwise the model would see one sentence in ten languages) |
| Model selection | Best of 5 epochs on the dev split of the *training* languages only. Zero-shot languages are never seen before test |
| Hardware | Apple M2, 8 GB, PyTorch MPS |

Three methods:

| Method | What trains | Trainable parameters |
|---|---|---:|
| Probe | Linear head only (encoder frozen) | 23,100 (0.02%) |
| **LoRA r = 8** | Rank-8 updates on attention query/value, α = 16, plus the head | **170,556 (0.14%)** |
| Full fine-tuning | Every transformer layer and the head | 21,662,652 (18.4%) |

"Full" freezes the word-embedding table. It holds 96M of the 118M parameters, and its Adam state alone
pushed an 8 GB machine into swap. A batch touches only a few hundred of its 250k rows, so every
transformer layer still trains.

## Results

Test accuracy (%), seed 0. Brackets on the mean gaps are paired bootstrap 95% CIs over test utterances.
The test sets are parallel, so utterance *k* is the same request in every language and every run.

| Run | en | de | fr | es | hi | ar | ja | zh | sw | ta | **mean** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Probe, en | 78.8 | 54.1 | 68.2 | 69.2 | 66.6 | 51.4 | 65.9 | 68.9 | 16.5 | 17.7 | 55.7 |
| **LoRA, en** | 87.1 | 68.7 | 79.5 | 76.7 | 74.9 | 59.1 | 74.6 | 77.3 | 23.8 | 47.9 | **67.0** |
| Full, en | 88.1 | 69.7 | 81.1 | 79.1 | 75.4 | 58.6 | 75.1 | 77.1 | 24.1 | 39.3 | 66.8 |
| Probe, multi | 75.1 | 59.1 | 68.9 | 69.3 | 66.8 | 54.3 | 67.7 | 69.1 | 22.8 | 31.3 | 58.4 |
| **LoRA, multi** | 83.9 | 78.5 | 81.4 | 81.2 | 79.0 | 70.0 | 82.1 | 81.2 | 58.7 | 73.3 | **76.9** |
| Full, multi | 86.1 | 81.4 | 83.9 | 83.1 | 82.1 | 73.1 | 83.6 | 83.1 | 65.6 | 77.1 | 79.9 |

| Comparison | Mean gap (points) |
|---|---|
| LoRA − full, English-only training | **+0.2** [−0.2, +0.6] |
| LoRA − full, multilingual training | **−3.0** [−3.4, −2.6] |
| LoRA − probe, English-only training | +11.2 [+10.5, +11.9] |
| LoRA multi − LoRA en | +10.0 [+9.4, +10.6] |
| Full multi − full en | +13.1 [+12.5, +13.8] |

**1. LoRA ties full fine-tuning when the target is narrow, and trails it when the target is broad.**
Trained on English, LoRA at 0.14% of the parameters is statistically indistinguishable from full
fine-tuning. Asked to fit ten languages at once, rank 8 leaves 3 points on the table. A higher rank is the
obvious next test.

**2. English-only training fails exactly where the pretraining is thinnest.** Swahili drops to 24% and
Tamil to 48% (LoRA). The same budget spread over ten languages lifts them to 59% and 73%, and costs
English 3 points. A frozen encoder can't use the multilingual data: the probe's Swahili only moves from
16.5% to 22.8%. The encoder itself has to adapt.

**3. English-only training breaks calibration in the same languages it breaks accuracy.**

| Expected calibration error (after temperature scaling) | en | hi | sw | ta |
|---|---:|---:|---:|---:|
| LoRA, en | 0.013 | 0.019 | 0.085 | 0.120 |
| LoRA, multi | 0.032 | 0.025 | 0.033 | 0.028 |

The temperature is fitted on English dev data, since that is all an English-only system has. It barely
moves the fine-tuned models (T ≈ 0.97–1.07), and it can't fix Swahili or Tamil, where the model is
confidently wrong. It matters a lot for the probe (mean ECE 0.22 → 0.07). Multilingual training makes
confidence uniformly reliable (0.02–0.03 everywhere).

**4. Handing off helps mid-resource languages, but can't rescue a language the model doesn't know.**
Accuracy on the requests the model keeps, when the least-confident 20% go to a human:

| | en | hi | sw | ta |
|---|---:|---:|---:|---:|
| LoRA, en: answer everything | 87.1 | 74.9 | 23.8 | 47.9 |
| LoRA, en: defer 20% | 94.8 | 84.8 | 28.1 | 55.9 |
| LoRA, multi: defer 20% | 92.7 | 88.6 | 68.0 | 83.7 |

In Hindi, deferring 20% buys 10 points. In Swahili it buys 4, because confidence there doesn't separate
right from wrong. Low-resource languages need training data, not a threshold.

## Limitations

- **One seed per run.** The paired CIs cover test-set sampling, not training variance. The LoRA-vs-full
  tie on English (+0.2 [−0.2, +0.6]) is the claim most sensitive to that.
- **Full fine-tuning freezes word embeddings** (see above). Updating them might widen its lead on
  multilingual data.
- **Why Swahili is weakest is untested.** How much Swahili the encoder saw in pretraining is not checked here.
- **One LoRA configuration** (r = 8 on query/value). No rank or target-module sweep.

## Reproduce

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
./setup.sh          # MASSIVE 1.1 (~40 MB compressed) + the encoder (~470 MB)
./run.sh            # 6 training runs (~2 h on an M2), then the evaluation
.venv/bin/pytest    # 12 unit tests
```

`results/*/logits.npz` holds every run's dev and test logits, so `python -m src.evaluate` reproduces
every table above in seconds without retraining.

## Layout

| File | What it does |
|---|---|
| `src/lora.py` | `LoRALinear` (W x + (α/r)·B A x, B initialised to zero) and `inject()` |
| `src/data.py` | MASSIVE loading and the equal-budget multilingual split |
| `src/train.py` | Probe / LoRA / full training, best-epoch selection, logit dumps |
| `src/evaluate.py` | Accuracy with bootstrap CIs, ECE before/after temperature scaling, abstention curves, paired gaps |
| `tests/` | LoRA starts exactly at the pretrained model, trains only A and B, scales correctly; ECE, temperature recovery, abstention and paired-gap checks; data split budget |
# abstain-or-route
