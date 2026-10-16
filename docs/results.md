<!-- GENERATED FILE - do not edit by hand.
     Produced by scripts/generate_docs.py from artifacts/results/experiments.json.
     Re-run that script after any training sweep. -->

# Results

_Generated 2026-10-16 10:42 UTC from `artifacts/results/experiments.json` (26 completed experiments)._

## Headline

| Result | Model | Protocol | Value |
|---|---|---|---|
| Best accuracy | ResNet50 + CBAM | benchmark | **99.14%** |
| Best macro F1 | ResNet50 + CBAM | benchmark | **0.9914** |
| Fastest inference | Logistic Regression (hand-crafted features) | classical | 0.02 ms/image |
| Smallest model | Ablation A: CNN | ablation | 4.8 MB |
| Best proposed (CBAM) model | EfficientNet-B0 + CBAM | production | 99.00% accuracy, 0.9898 macro F1 |
| Serving in the application | EfficientNet-B0 | production | selected by `f1_macro` |
| Evaluated on | held-out test split | — | 17,572 images, 38 classes |

Rows with different protocols are **not** comparable: `benchmark` models were trained on a fixed subset and scored on a 4,560-image split, `production` models on the full data and scored on 17,572 images.

★ marks the proposed CBAM architectures.


## Dataset

Source: Kaggle `vipoooool/new-plant-diseases-dataset` (PlantVillage, augmented). 38 classes across 14 plant species.

| Split | Images | Classes | Smallest class | Largest class | Imbalance |
|---|---:|---:|---:|---:|---:|
| train | 70,295 | 38 | 1,642 | 2,022 | 1.23× |
| valid | 17,572 | 38 | 410 | 505 | 1.23× |

- **Healthy classes:** 12 · **disease classes:** 26
- **Plants:** Apple; Blueberry; Cherry (including sour); Corn (maize); Grape; Orange; Peach; Pepper, bell; Potato; Raspberry; Soybean; Squash; Strawberry; Tomato
- **Duplicate / leakage check** (perceptual hash (pHash, 8x8 DCT), exact-hash collisions only, 60 images sampled per class): 0 duplicates in train, 0 in valid, 0 train/valid collisions.

The dataset's own `valid` folder is used as the held-out **test** split; validation for early stopping and calibration is carved out of `train` (see `data/splits.json`), so no test image influences training or model selection.


## The deployed model: EfficientNet-B0

Every metric for the model the application serves (`efficientnet_b0__prod__9a259f82`), on the full held-out test split.

| Metric | Value |
|---|---:|
| Test images | 17,572 |
| Accuracy (top-1) | 99.06% |
| Top-3 accuracy | 99.94% |
| Top-5 accuracy | 99.98% |
| Balanced accuracy | 99.04% |
| Macro precision | 0.9906 |
| Macro recall | 0.9904 |
| Macro F1 | 0.9904 |
| Weighted F1 | 0.9905 |
| Cohen's kappa | 0.9903 |
| Matthews correlation | 0.9903 |
| ROC AUC (macro, one-vs-rest) | 0.99997 |
| Expected calibration error (raw) | 0.0500 |
| Brier score | 0.0199 |
| Mean confidence when right / wrong | 94.4% / 56.3% |
| Inference, batch 1 | 18.50 ms (cuda) |

166 of 17,572 test images are misclassified.


## Per-class accuracy (deployed model)

Sorted hardest first. Recall is the per-class accuracy: the share of that disease's test images the model gets right.

| # | Class | Recall (accuracy) | Precision | F1 | Test images |
|---:|---|---:|---:|---:|---:|
| 1 | Tomato - Target Spot | 89.28% | 97.84% | 0.9336 | 457 |
| 2 | Corn (maize) - Cercospora leaf spot Gray leaf spot | 93.90% | 96.98% | 0.9542 | 410 |
| 3 | Corn (maize) - Northern Leaf Blight | 97.27% | 94.50% | 0.9587 | 477 |
| 4 | Tomato - Spider mites Two-spotted spider mite | 99.77% | 93.74% | 0.9666 | 435 |
| 5 | Tomato - Early blight | 96.88% | 98.10% | 0.9748 | 480 |
| 6 | Tomato - Septoria leaf spot | 97.94% | 97.94% | 0.9794 | 436 |
| 7 | Tomato - Late blight | 97.19% | 99.12% | 0.9815 | 463 |
| 8 | Tomato - Healthy | 100.00% | 96.78% | 0.9836 | 481 |
| 9 | Tomato - Bacterial spot | 98.59% | 98.59% | 0.9859 | 425 |
| 10 | Tomato - Tomato Yellow Leaf Curl Virus | 99.80% | 98.39% | 0.9909 | 490 |
| 11 | Soybean - Healthy | 99.80% | 98.63% | 0.9921 | 505 |
| 12 | Potato - Late blight | 99.59% | 98.98% | 0.9928 | 485 |
| 13 | Apple - Healthy | 98.80% | 100.00% | 0.9940 | 502 |
| 14 | Peach - Healthy | 100.00% | 98.86% | 0.9942 | 432 |
| 15 | Peach - Bacterial spot | 98.91% | 100.00% | 0.9945 | 459 |
| 16 | Grape - Black rot | 99.15% | 99.79% | 0.9947 | 472 |
| 17 | Grape - Esca (Black Measles) | 99.79% | 99.17% | 0.9948 | 480 |
| 18 | Tomato - Leaf Mold | 99.15% | 100.00% | 0.9957 | 470 |
| 19 | Blueberry - Healthy | 99.78% | 99.56% | 0.9967 | 454 |
| 20 | Potato - Healthy | 99.56% | 99.78% | 0.9967 | 456 |
| 21 | Pepper, bell - Healthy | 99.60% | 99.80% | 0.9970 | 497 |
| 22 | Apple - Cedar apple rust | 99.77% | 99.77% | 0.9977 | 440 |
| 23 | Pepper, bell - Bacterial spot | 99.58% | 100.00% | 0.9979 | 478 |
| 24 | Potato - Early blight | 100.00% | 99.59% | 0.9979 | 485 |
| 25 | Apple - Black rot | 100.00% | 99.60% | 0.9980 | 497 |
| 26 | Orange - Haunglongbing (Citrus greening) | 100.00% | 99.60% | 0.9980 | 503 |
| 27 | Cherry (including sour) - Powdery mildew | 100.00% | 99.76% | 0.9988 | 421 |
| 28 | Grape - Healthy | 99.76% | 100.00% | 0.9988 | 423 |
| 29 | Strawberry - Leaf scorch | 99.77% | 100.00% | 0.9989 | 444 |
| 30 | Tomato - Tomato mosaic virus | 100.00% | 99.78% | 0.9989 | 448 |
| 31 | Cherry (including sour) - Healthy | 99.78% | 100.00% | 0.9989 | 456 |
| 32 | Corn (maize) - Healthy | 100.00% | 99.79% | 0.9989 | 465 |
| 33 | Apple - Apple scab | 100.00% | 99.80% | 0.9990 | 504 |
| 34 | Corn (maize) - Common rust | 100.00% | 100.00% | 1.0000 | 477 |
| 35 | Grape - Leaf blight (Isariopsis Leaf Spot) | 100.00% | 100.00% | 1.0000 | 430 |
| 36 | Raspberry - Healthy | 100.00% | 100.00% | 1.0000 | 445 |
| 37 | Squash - Powdery mildew | 100.00% | 100.00% | 1.0000 | 434 |
| 38 | Strawberry - Healthy | 100.00% | 100.00% | 1.0000 | 456 |

14 of 38 classes are recognised with 100% recall; the hardest is **Tomato - Target Spot** at 89.28%.

## Benchmark

Every model in the `benchmark` protocol was trained on an identical fixed-budget subset (300 train / 60 val / 120 test images per class, 160 px, 12 epochs, identical optimiser and seed). Only the architecture differs. Full-data `production` runs are in their own section below and are **not** directly comparable with these subset rows.

| | Model | Protocol | Accuracy | Precision | Recall | Macro F1 | Top-3 | Params | Size | ms/img | ECE |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|  | ResNet50 + CBAM | benchmark | 99.14% | 0.9917 | 0.9914 | 0.9914 | 99.96% | 24.11M | 92.2 MB | 15.93 | 0.0612 |
|  | ResNet50 | benchmark | 99.12% | 0.9914 | 0.9912 | 0.9912 | 99.93% | 23.59M | 90.2 MB | 14.19 | 0.0614 |
|  | DenseNet121 | benchmark | 97.81% | 0.9781 | 0.9781 | 0.9780 | 99.78% | 6.99M | 27.0 MB | 33.52 | 0.0617 |
|  | EfficientNet-B0 | benchmark | 97.02% | 0.9707 | 0.9702 | 0.9700 | 99.65% | 4.06M | 15.6 MB | 19.47 | 0.0721 |
| **★** | EfficientNet-B0 + CBAM | benchmark | 96.82% | 0.9688 | 0.9682 | 0.9681 | 99.63% | 4.26M | 16.4 MB | 17.84 | 0.0727 |
| **★** | Custom CNN + CBAM | benchmark | 96.07% | 0.9612 | 0.9607 | 0.9605 | 99.58% | 1.26M | 4.8 MB | 7.39 | 0.0418 |
|  | Custom CNN + SE | benchmark | 96.03% | 0.9609 | 0.9603 | 0.9598 | 99.54% | 1.26M | 4.8 MB | 4.47 | 0.0626 |
|  | Custom CNN (baseline) | benchmark | 95.83% | 0.9586 | 0.9583 | 0.9578 | 99.50% | 1.25M | 4.8 MB | 2.13 | 0.0670 |
|  | MobileNetV3-Large | benchmark | 95.02% | 0.9507 | 0.9502 | 0.9497 | 99.25% | 3.01M | 11.6 MB | 13.64 | 0.0957 |
|  | MobileNetV3-Large + CBAM | benchmark | 94.78% | 0.9480 | 0.9478 | 0.9473 | 99.14% | 3.12M | 12.0 MB | 14.24 | 0.0901 |
|  | XGBoost (hand-crafted features) | classical | 93.07% | 0.9309 | 0.9307 | 0.9302 | 98.86% | — | — | 0.05 | — |
|  | Random Forest (hand-crafted features) | classical | 90.53% | 0.9068 | 0.9053 | 0.9037 | 97.65% | — | — | 0.09 | — |
|  | Linear SVM (hand-crafted features) | classical | 87.96% | 0.8785 | 0.8796 | 0.8778 | 96.01% | — | — | 0.09 | — |
|  | Logistic Regression (hand-crafted features) | classical | 87.24% | 0.8726 | 0.8724 | 0.8719 | 97.15% | — | — | 0.02 | — |


## Full-data production runs

| | Model | Protocol | Accuracy | Precision | Recall | Macro F1 | Top-3 | Params | Size | ms/img | ECE |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|  | EfficientNet-B0 | production | 99.06% | 0.9906 | 0.9904 | 0.9904 | 99.94% | 4.06M | 15.6 MB | 18.50 | 0.0500 |
| **★** | EfficientNet-B0 + CBAM | production | 99.00% | 0.9900 | 0.9898 | 0.9898 | 99.94% | 4.26M | 16.4 MB | 13.17 | 0.0500 |
|  | CBAM on last two stages | production | 98.81% | 0.9881 | 0.9881 | 0.9880 | 99.94% | 1.26M | 4.8 MB | 4.18 | 0.0275 |
| **★** | Custom CNN + CBAM | production | 98.42% | 0.9845 | 0.9842 | 0.9842 | 99.90% | 1.26M | 4.8 MB | 7.41 | 0.0245 |


## Attention ablation

**Research question:** does integrating channel and spatial attention through CBAM improve plant disease classification accuracy, robustness and interpretability compared with conventional CNN architectures?

| Arm | Configuration | Accuracy | Macro F1 | Macro recall | Δ Macro F1 | Δ Accuracy | Params | ms/img |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **A** | CNN (control) | 95.88% | 0.9582 | 0.9588 | — | — | 1,248,774 | 1.92 |
| **B** | CNN + Channel attention | 96.34% | 0.9631 | 0.9634 | +0.49 pt | +0.46 pt | 1,259,782 | 4.22 |
| **C** | CNN + Spatial attention | 96.12% | 0.9606 | 0.9612 | +0.24 pt | +0.24 pt | 1,249,166 | 3.25 |
| **D** | CNN + SE | 96.38% | 0.9634 | 0.9638 | +0.51 pt | +0.50 pt | 1,259,782 | 5.37 |
| **E** | CNN + CBAM | 96.29% | 0.9628 | 0.9629 | +0.45 pt | +0.42 pt | 1,260,174 | 6.66 |


**Answer from the data:** CBAM improved macro F1 by 0.45 points, which is within the 0.47-point run-to-run variation measured on this setup. The improvement is directionally positive but not established.


> All five arms share one architecture, one training budget, one data subset and one seed; only the attention block differs. A single run per arm cannot establish statistical significance. Retraining cnn_baseline at 3 seeds moved macro F1 across 0.22 points (SD 0.11). Repeating identical configurations moved it further still, by up to 0.47 points, so 0.47 points is used as the threshold: it is the largest variation observed between runs that should have been equivalent. A gap between attention arms smaller than this is not evidence of an architectural effect.


## Confidence calibration

Temperature scaling fits one scalar on the validation split. It cannot change which class wins, so accuracy is unchanged by construction; only the confidence values move.

| Model | ECE (raw) | ECE (scaled) | Temperature | Improvement |
|---|---:|---:|---:|---:|
| Custom CNN + CBAM | 0.0245 | 0.0020 | 0.689 | +92% |
| CBAM on last two stages | 0.0275 | 0.0018 | 0.648 | +93% |
| Ablation E: CNN + CBAM | 0.0413 | 0.0043 | 0.647 | +89% |
| Custom CNN + CBAM | 0.0418 | 0.0046 | 0.667 | +89% |
| Ablation B: CNN + Channel attention | 0.0429 | 0.0036 | 0.642 | +92% |
| CBAM on last two stages | 0.0429 | 0.0051 | 0.662 | +88% |
| CBAM on every stage | 0.0472 | 0.0057 | 0.670 | +88% |
| EfficientNet-B0 + CBAM | 0.0500 | 0.0023 | 0.520 | +95% |
| EfficientNet-B0 | 0.0500 | 0.0022 | 0.507 | +96% |
| Ablation D: CNN + SE | 0.0582 | 0.0083 | 0.644 | +86% |
| ResNet50 + CBAM | 0.0612 | 0.0022 | 0.604 | +96% |
| ResNet50 | 0.0614 | 0.0019 | 0.581 | +97% |
| DenseNet121 | 0.0617 | 0.0055 | 0.488 | +91% |
| Custom CNN + SE | 0.0626 | 0.0063 | 0.619 | +90% |
| Ablation C: CNN + Spatial attention | 0.0657 | 0.0077 | 0.618 | +88% |
| Custom CNN (baseline) | 0.0670 | 0.0064 | 0.598 | +91% |
| Ablation A: CNN | 0.0677 | 0.0063 | 0.597 | +91% |
| CBAM on last stage only | 0.0677 | 0.0060 | 0.560 | +91% |
| EfficientNet-B0 | 0.0721 | 0.0046 | 0.561 | +94% |
| EfficientNet-B0 + CBAM | 0.0727 | 0.0025 | 0.583 | +97% |
| MobileNetV3-Large + CBAM | 0.0901 | 0.0074 | 0.566 | +92% |
| MobileNetV3-Large | 0.0957 | 0.0080 | 0.559 | +92% |


## Out-of-distribution handling

Calibrated on 4000 held-out test images against 1200 synthetic_proxies samples, at a 95% target true-positive rate.

| Score | AUROC | Threshold | OOD caught |
|---|---:|---:|---:|
| Max softmax probability | 0.9752 | 0.1500 | 93.4% |
| Normalised entropy | 0.9770 | 0.8500 | 93.2% |
| Free energy | 0.9415 | -4.6246 | 72.8% |


### Whole-pipeline rejection

The image-quality gate runs *before* the model, so the end-to-end rate is what a user actually experiences.

| Stage | OOD rejected |
|---|---:|
| Image-quality gate alone | 63.0% |
| OOD score alone | 93.5% |
| **Combined pipeline** | **95.0%** |
| Cost: genuine leaves rejected | 2.0% |


| OOD family | Quality gate | OOD score | Combined |
|---|---:|---:|---:|
| extreme | 51.0% | 81.0% | 87.0% |
| flat | 100.0% | 100.0% | 100.0% |
| noise | 100.0% | 100.0% | 100.0% |
| shuffle | 1.0% | 93.0% | 93.0% |


> **Caveat.** Synthetic proxies (uniform noise, flat colour fields, tile-shuffled leaves, heavily degraded leaves) are genuinely out of distribution but are easier to reject than real photographs of unrelated subjects. The detection rates below are therefore an upper bound. Re-run with --ood-dir pointing at a real non-leaf corpus for a tighter estimate.


## Classical baselines

Trained on 1184 hand-crafted features (colour histograms and moments, LBP, GLCM, HOG) extracted from 11,400 images.


**Not trained:** RBF SVM. 
The RBF SVM is skipped by default: `probability=True` fits five internal cross-validation folds of an O(n^2) solver over 11,400 samples, which costs hours on this hardware for a baseline the linear SVM already covers. Run it explicitly with `python training/train_classical.py --models rbf_svm` if the comparison is wanted.


## Accuracy on field-like photos

Every number above comes from laboratory photographs. This measures how the full pipeline (quality gate, dust check, model, OOD check) behaves when the same leaves are made to look like field photos - 30 disease images per condition, produced by `scripts/domain_shift_probe.py`.

| Condition | Correct class | vs lab | Not marked reliable |
|---|---:|---:|---:|
| Original lab photo | 96.7% (29/30) | — | 6.7% |
| Leaf on a soil background | 100.0% (30/30) | +3.3 pt | 3.3% |
| Photographed off-axis | 96.7% (29/30) | +0.0 pt | 3.3% |
| Leaf small in the frame (far away) | 10.0% (3/30) | -86.7 pt | 73.3% |
| Harsh directional sunlight | 96.7% (29/30) | +0.0 pt | 3.3% |
| Heavy JPEG compression | 90.0% (27/30) | -6.7 pt | 10.0% |
| All of the above (closest to a field photo) | 76.7% (23/30) | -20.0 pt | 20.0% |

*Correct class* counts the top prediction even when the app flagged it; *not marked reliable* is any outcome other than a plain `ok` - a refusal, a low-confidence warning or a provisional result.

A small leaf far away in the frame is the dominant failure: the model was trained on leaves that fill the image, so framing matters more than lighting, angle or compression. The app's advice for such photos is to move closer.

> Transformations imitate field conditions applied to laboratory images. Real photographs additionally bring occlusion, motion blur, overlapping leaves and unseen species, so this is a lower bound on the accuracy drop rather than an estimate of it.


## Dust, soil and sand detection

PlantVillage has no "dusty leaf" class, so a dust film used to be classified as a disease (often powdery mildew) or could hide one. A small detector in `backend/app/ml/dust.py` flags it before diagnosis. Evaluated on 3,040 held-out images (AUC **0.968**; 0.913 on crops withheld from training: Squash, Cherry_(including_sour), Corn_(maize), Grape).

| Image | n | Flagged "possible dust" (provisional) | Flagged "heavy dust" (withheld) |
|---|---:|---:|---:|
| Clean leaf (lab photo) | 760 | 0.1% | 0.0% |
| Clean leaf, field-like conditions | 760 | 5.9% | 1.1% |
| Light dust | 211 | 73.5% | 50.7% |
| Moderate dust | 250 | 87.2% | 71.6% |
| Heavy dust | 299 | 93.0% | 84.0% |
| Light dust, field-like | 189 | 50.8% | 29.1% |
| Moderate dust, field-like | 243 | 65.4% | 43.6% |
| Heavy dust, field-like | 328 | 83.8% | 61.3% |

Rows marked *clean* are false alarms. Genuine powdery-mildew leaves (n=80) trip the possible-dust warning 11.2% of the time and the heavy-dust refusal 2.5%, because a pale fungal coating and dust look alike; that is why the warning says so instead of refusing.

> Positives are synthetic dust films on real leaves; no dataset of photographed dusty leaves exists. Rates are on held-out images.


## Hyperparameter search

Random search over `cnn_cbam`, 6 trials of 8 epochs, ranked by best validation accuracy.

| Rank | learning_rate | weight_decay | dropout | Val accuracy | Val loss | Minutes |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.001 | 1e-05 | 0.2 | 96.49% | 0.5702 | 14 |
| 2 | 0.001 | 0.001 | 0.2 | 95.92% | 0.5808 | 12 |
| 3 | 0.001 | 0.0001 | 0.2 | 95.88% | 0.5704 | 12 |
| 4 | 0.001 | 1e-05 | 0.4 | 94.25% | 0.6308 | 12 |
| 5 | 0.0001 | 0.001 | 0.2 | 90.04% | 0.7502 | 12 |
| 6 | 0.0001 | 1e-05 | 0.2 | 89.96% | 0.7606 | 12 |

Best: learning_rate=0.001, weight_decay=1e-05, dropout=0.2 at 96.49% validation accuracy.
The learning rate dominates: every trial at 0.001 beats every trial at any other rate within this short budget.


## Serving latency

Measured through the running API with `scripts/benchmark_serving.py` on 2026-08-31: EfficientNet-B0 on cuda, 40 requests. Checks added to the quality gate after that date are not included; re-run the script to refresh.

| Stage | Median | p95 |
|---|---:|---:|
| Image-quality checks | 2.7 ms | 3.7 ms |
| Preprocessing | 2.0 ms | 2.8 ms |
| Model inference | 17.1 ms | 21.4 ms |
| Explanation heat maps | 181.6 ms | 265.1 ms |
| **Total request** | 205.1 ms | 290.1 ms |

Throughput: 4.91 requests/s with explanations on. The free Render deployment runs the same model through ONNX Runtime on a shared CPU, so expect slower responses there.


## Reproducing these numbers

Every stage is idempotent and resumable, so the whole sequence runs with one command:

```bash
python scripts/run_pipeline.py
```

or step by step:

```bash
python scripts/probe_hardware.py
python scripts/download_dataset.py
#   add --path /path/to/dataset to use a copy you already have;
#   this is the only stage that can need Kaggle credentials
python training/dataset_inspect.py
python training/prepare_splits.py
python scripts/build_disease_info.py
python training/train_classical.py
python training/run_experiments.py --suite research
python training/run_experiments.py --suite production
python training/analyse_results.py
python training/export_model.py --criterion f1_macro --torchscript --onnx
python training/calibrate_ood.py
python scripts/benchmark_serving.py
python scripts/generate_docs.py
```

On a memory-constrained machine, wrap the training sweep in the supervisor: it relaunches the suite if the process is killed and skips whatever is already in the ledger.

```bash
python scripts/supervise_training.py --suite research
```
