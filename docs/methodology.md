# Methodology

This document states exactly what was done, in enough detail to reproduce it,
and is explicit about the compromises the available hardware forced.

---

## 1. Dataset

**Source:** [`vipoooool/new-plant-diseases-dataset`](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
on Kaggle — an augmented derivative of the PlantVillage corpus.

Nothing about the archive's layout was assumed. `training/dataset_inspect.py`
walks the download, scores every directory that looks like an ImageFolder split,
and derives class names, counts, geometry and corruption rates from the files
actually present. The measured result is written to
`artifacts/dataset_report.json`:

| Property | Measured value |
|---|---|
| Official `train/` | 70,295 images |
| Official `valid/` | 17,572 images |
| Classes | 38 |
| Distinct plant species | 14 |
| Healthy classes / diseased classes | 12 / 26 |
| Image dimensions | 256 × 256, uniformly |
| Corrupt images (1,520 sampled) | 0 |
| Imbalance ratio (max class / min class) | 1.23 |
| Coefficient of variation of class sizes | 0.056 |
| Unlabelled extras | `test/test/` — 33 loose images, excluded |

The archive nests the real data two directories deep and duplicates the folder
name; the discovery step handles that without hard-coded paths.

### Duplicate and leakage check

The dataset ships pre-augmented, which raises an obvious question: do augmented
copies of the same original photograph appear on both sides of the official
train/valid boundary? If so, the reported accuracy is inflated.

An exact all-pairs comparison over 87,867 images was not affordable here, so a
**perceptual hash (pHash, 8×8 DCT)** was computed for a stratified sample of 60
images per class per split and collisions were counted:

| Check | Result |
|---|---|
| Duplicate groups within train (2,280 sampled) | 0 |
| Duplicate groups within valid | 0 |
| Train ↔ valid hash collisions | **0** (0.00% of sampled valid images) |

This is a *sampled* estimate and is reported as such. It gives no evidence of
leakage; it is not proof of its absence. pHash also catches only near-identical
images — it would not flag a rotated or flipped augmentation of the same
original, which is a real limitation of this check.

---

## 2. Split protocol

```
official train/ (70,295)  ──stratified 90/10──►  train  63,265
                                             └►  val     7,030   (early stopping, model selection)

official valid/ (17,572)  ─────────────────────►  test  17,572   (reported numbers only)
```

The official `valid/` directory is treated as the **test** set and is never used
for early stopping, learning-rate scheduling, checkpoint selection or
temperature fitting. Validation is carved out of the official training data
instead. That gives a genuine three-way split in which nothing that influences
model selection is later scored as test data.

Both splits are stratified: every class keeps its proportion, and the splitter
guarantees at least one sample per class on each side.

Splits are materialised once into `data/manifests/*.csv` and every downstream
script — classical baselines, deep benchmark, ablation — reads those files. No
script re-shuffles the data, so all experiments see identical rows in identical
order.

---

## 3. Class imbalance

The measured imbalance ratio is **1.23** with a coefficient of variation of
**0.056** — the dataset is close to uniform. `prepare_splits.py` applies this
decision rule to the real counts:

| Condition | Strategy chosen |
|---|---|
| ratio < 1.5 and CV < 0.20 | none (plain cross-entropy) |
| ratio < 4.0 | inverse-square-root class weights |
| ratio < 15.0 | inverse-frequency class weights |
| otherwise | class-balanced sampler |

This dataset lands in the first band, so **no imbalance correction is applied**,
and the reason is stored in `data/splits.json`:

> Imbalance ratio 1.23 and coefficient of variation 0.056 are both small.
> Re-weighting a near-uniform distribution adds gradient noise without improving
> minority-class recall, so plain cross-entropy is used.

Focal loss, class weighting and balanced sampling are all implemented and
selectable; they are simply not warranted here. The brief asked not to stack
every technique at once, and stacking them on a balanced dataset would be
cargo-culting.

---

## 4. Preprocessing and augmentation

**Inference and validation** (deterministic, `build_eval_transform`):

```
Resize(shorter side → 1.14 × S) → CenterCrop(S) → ToTensor → Normalize(ImageNet)
```

Resize-then-crop preserves aspect ratio, matching how the ImageNet backbones
were trained. A direct squash to S×S would distort leaf shape.

**Training only** (`build_train_transform`), three presets:

| Transform | light | medium | strong |
|---|---|---|---|
| RandomResizedCrop scale | 0.85–1.0 | 0.7–1.0 | 0.55–1.0 |
| Horizontal flip | 0.5 | 0.5 | 0.5 |
| Vertical flip | 0.3 | 0.3 | 0.3 |
| Rotation | ±15° | ±25° | ±35° |
| Colour jitter (brightness/contrast/sat) | 0.12 | 0.22 | 0.32 |
| Hue jitter | 0.018 | 0.033 | 0.048 |
| Gaussian blur | — | — | p=0.2 |
| Random erasing | — | — | p=0.15 |

Design constraints applied:

* **Only biologically plausible transforms.** Leaves are near-symmetric and are
  photographed from arbitrary angles, so flips and rotations are genuine
  viewpoint changes rather than impossible images.
* **Hue jitter is deliberately small.** Hue is the signal for chlorosis and
  necrosis. Shifting it aggressively would train the model to ignore the very
  cue a plant pathologist uses.
* **Augmentation is applied to the training split only.** Validation and test go
  through the deterministic pipeline, so evaluation measures the model, not the
  augmentation.

The same module is imported by the training scripts *and* by the serving code,
which is what guarantees an uploaded image is normalised exactly as training
images were.

---

## 5. Hardware and the compromises it forced

Measured at run time by `training/config.py`:

| Resource | Value |
|---|---|
| GPU | NVIDIA GeForce GTX 1650, 4 GB VRAM |
| System RAM | 7.35 GB |
| CPU | AMD Ryzen 5 5600H, 6 cores / 12 threads |
| PyTorch / CUDA | 2.13.0 + cu126 |

Three adaptations follow from this, each measured rather than assumed:

**Mixed precision is disabled.** AMP is normally free speed, but only on
hardware with tensor cores. A micro-benchmark run at startup measured fp16 at
**0.24×** the speed of fp32 on this card (the GTX 1650's TU117 has no tensor
cores and its fp16 convolutions take a slow path). Enabling AMP would have
roughly tripled the total training time. The measured ratio is cached in
`models/metadata/amp_probe.json`, and AMP is enabled only when the measured
speed-up exceeds 1.15×.

The full measurement is reproducible — raw matrix-multiply TFLOPS at both
precisions plus per-architecture training-step throughput:

```bash
python scripts/probe_hardware.py
```

which writes `artifacts/results/hardware_probe.json`. On the development machine
it reports fp32 at 1.63 TFLOPS against fp16 at 0.34 TFLOPS (a ratio of 0.21) and
a median AMP speed-up of 0.35× across architectures.

**Batch sizes are capped per architecture.** A table of VRAM ceilings scaled by
input area, halved when AMP is off, keeps ResNet50 and DenseNet121 inside 4 GB.
A CUDA OOM still triggers one automatic retry at half the batch size.

**DataLoader workers are budgeted against the commit limit.** On Windows each
spawned worker re-imports the CUDA build of torch and reserves ~1 GB of commit
charge. Exceeding the system limit produces `WinError 1455` and a run that hangs
on workers that never started — which happened during development. The worker
count is now budgeted against physical RAM, and a spawn failure is detected and
retried with in-process loading rather than hanging.

### The benchmark subset, and why it exists

Training 18 architectures on 63,265 images was not feasible on a 4 GB card in
the time available. Rather than train each model on whatever budget happened to
be convenient — which would make the comparison meaningless — every model in the
benchmark, the ablation and the placement study is trained on an **identical
fixed-budget subset** under an **identical schedule**:

| Split | Per class | Total |
|---|---|---|
| train_bench | 300 | 11,400 |
| val_bench | 60 | 2,280 |
| test_bench | 120 | 4,560 |

with image size 160, 12 epochs, light augmentation, AdamW, cosine schedule with
one warm-up epoch, and early stopping at patience 4. **Only the architecture
changes.** That is what makes the resulting table a controlled comparison rather
than a leaderboard of unrelated runs.

Selected architectures are then retrained on the **full** training split at
224 px for the production model, over 6 epochs with early stopping at patience 2.
Six rather than twelve is not a shortcut: the full split carries 1,660 images per
class against the subset's 300, so each epoch delivers 5.5× the gradient signal
and the run reaches its plateau in far fewer passes. The epoch budget and the
epoch actually selected are both recorded in the ledger for every run.

The `protocol` column in the benchmark table says which regime produced each row,
and the two are never compared as though they were the same experiment.

---

## 6. Training procedure

Common to every run:

| Setting | Value |
|---|---|
| Optimiser | AdamW |
| Base learning rate | 3 × 10⁻⁴ |
| Weight decay | 1 × 10⁻⁴ |
| Schedule | Linear warm-up (1 epoch) → cosine decay to 1% of peak |
| Loss | Cross-entropy, label smoothing 0.05 |
| Gradient clipping | Global norm 1.0 |
| Model selection | Best validation accuracy, ties broken by validation loss |
| Early stopping | Patience 4 (benchmark) / 3 (production) |
| Seed | 42, applied to Python, NumPy, torch and every DataLoader worker |

### Two-stage transfer learning

Pretrained backbones are trained in two stages:

**Stage 1 (epochs 1–3): backbone frozen.** Only the randomly initialised head
trains. A random head produces large, badly directed gradients; letting them
flow into pretrained weights on the first step destroys the features that made
the backbone worth using.

**Stage 2 (epoch 4 onwards): top 3 blocks unfrozen, learning rate × 0.15.** The
lower layers keep their generic ImageNet edge and texture filters; the upper,
more semantic layers adapt to leaf pathology. The reduced learning rate nudges
those weights rather than overwriting them. The optimiser and scheduler are
rebuilt at the transition, and the boundary is marked on every training curve.

### Hyper-parameter search

A full sweep is not affordable on this hardware - roughly ten minutes per
configuration - so the search is deliberately coarse and is reported as such.
`training/tune_hyperparameters.py` searches the three settings that actually
move the result for AdamW on this problem:

| Parameter | Values searched |
|---|---|
| Learning rate | 1e-4, 3e-4, 1e-3 |
| Weight decay | 1e-5, 1e-4, 1e-3 |
| Dropout | 0.2, 0.4 |

Random search rather than a grid, following Bergstra & Bengio (2012): when only
a few parameters matter, random sampling covers the important axes better than a
grid of the same size, because a grid wastes trials repeating identical values
along the unimportant axes.

Trials run on the benchmark subset at a reduced epoch budget and are selected on
**validation** accuracy - the test split is not touched. Results land in
`artifacts/results/tuning_results.json` under the `tuning` group and are never
mixed into the benchmark table.

Two things were **not** searched, and the reason matters:

* **Image size and batch size** are pinned by the protocol. Varying them would
  change the compute budget between arms and break the controlled comparison.
* **Number of frozen/fine-tuned layers** is fixed at "top 3 blocks" across every
  transfer model for the same reason. The CBAM *placement* study is the one
  architectural search that is run, because it answers a question about the
  proposed contribution rather than about optimisation.

```bash
python training/tune_hyperparameters.py --strategy random --trials 6
```

### Reproducibility

`set_seed` seeds Python's `random`, NumPy, torch CPU and CUDA generators, and
gives each DataLoader worker a distinct derived seed. cuDNN autotuning is left
**on** by default: forcing full determinism costs roughly 15–20% throughput, and
the conclusions drawn here rest on differences far larger than the residual
non-determinism. `--deterministic` is available when bit-exactness matters.

---

## 7. Evaluation

Every model is evaluated on the held-out test split and produces:

* accuracy, balanced accuracy, macro/weighted/micro precision, recall and F1
* top-1, top-3, top-5 accuracy
* one-vs-rest ROC-AUC (macro and weighted)
* Cohen's κ and Matthews correlation coefficient
* per-class precision, recall, F1 and support
* a confusion matrix, row-normalised
* expected and maximum calibration error, Brier score, reliability diagram
* single-image and batched inference latency (mean, median, p95)
* parameter count and on-disk model size

### Confidence calibration

Raw softmax outputs are not probabilities you can act on. **Temperature scaling**
(Guo et al., 2017) fits a single scalar T on the *validation* split — never the
test split — and divides the logits by it. Because a scalar divisor cannot
change which logit is largest, accuracy is provably unchanged; only the
confidence values move. Both the raw and scaled reliability diagrams are
rendered so the improvement is visible rather than asserted.

### Out-of-distribution handling

Three scores are computed on the model's own outputs: maximum softmax
probability, normalised predictive entropy, and free energy (Liu et al., 2020).
Thresholds are chosen to retain a target **95% true-positive rate** on genuine
test leaves, so the user-visible behaviour is predictable: roughly 5% of real
leaf photographs will be asked for a retake.

The OOD set is built from synthetic proxies — uniform noise, flat colour fields,
tile-shuffled leaves and heavily degraded leaves — because no external non-leaf
corpus ships with this project. These are genuinely out of distribution but
**easier to reject than a real photograph of an unrelated subject**, so the
detection rates are an upper bound. `--ood-dir` accepts a real corpus. The
caveat is stored in the JSON so it travels with the numbers.

Crucially, the OOD score never runs alone: the image-quality gate runs first, so
the reported figure is the **end-to-end pipeline** rejection rate, which is what
a user actually experiences.

---

## 8. What this methodology does not establish

Stated plainly, because a final-year project should be honest about its limits:

* **One run per configuration.** Each architecture was trained once, so no
  delta can be given a confidence interval. The threshold for calling a
  difference real is measured rather than assumed — see the noise floor section
  of [`ablation_study.md`](ablation_study.md) — but a measured floor is still not
  a significance test.

  Two distinct floors matter here and they are not interchangeable. Repeating an
  *identical* configuration bounds cuDNN scheduling jitter only; it holds the
  initialisation fixed, which the arms being compared do not, because different
  attention blocks have different parameter shapes and so draw different starting
  weights from the same seed. The wider floor — one architecture retrained at
  several seeds — is the one that licenses architectural claims. Where it has not
  been measured, the generated report says so and marks its conclusions as
  correspondingly weaker.
* **Benchmark subset.** The comparison table is fair *within itself* but its
  absolute numbers are lower than full-data training would give. Rankings can
  shift with more data — small models are favoured in low-data regimes.
* **Laboratory imagery.** PlantVillage images are single leaves on uniform
  backgrounds under controlled lighting. Accuracy on this test split says little
  about performance on a field photograph with soil, sky, multiple leaves and
  harsh sun. This is the single largest gap between this project and a
  deployable agricultural tool.
* **pHash leakage check is sampled and rotation-blind.**
* **38 classes, 14 species.** Anything outside that set can only be rejected,
  never identified — which is precisely why the OOD stage exists.
