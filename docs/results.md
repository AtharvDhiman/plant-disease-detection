<!-- GENERATED FILE - do not edit by hand.
     Produced by scripts/generate_docs.py from artifacts/results/experiments.json.
     Re-run that script after any training sweep. -->

# Results

_Generated 2026-09-01 07:05 UTC from `artifacts/results/experiments.json` (26 completed experiments)._

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

## Benchmark

Every model in the `benchmark` protocol was trained on an identical fixed-budget subset (300 train / 60 val / 120 test images per class, 160 px, 12 epochs, identical optimiser and seed). Only the architecture differs. Rows marked `production` were trained on the full training split at 224 px and are **not** directly comparable with the subset rows.

| | Model | Protocol | Accuracy | Precision | Recall | Macro F1 | Top-3 | Params | Size | ms/img | ECE |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|  | ResNet50 + CBAM | benchmark | 99.14% | 0.9917 | 0.9914 | 0.9914 | 99.96% | 24.11M | 92.2 MB | 15.93 | 0.0612 |
|  | ResNet50 | benchmark | 99.12% | 0.9914 | 0.9912 | 0.9912 | 99.93% | 23.59M | 90.2 MB | 14.19 | 0.0614 |
|  | EfficientNet-B0 | production | 99.06% | 0.9906 | 0.9904 | 0.9904 | 99.94% | 4.06M | 15.6 MB | 18.50 | 0.0500 |
| **★** | EfficientNet-B0 + CBAM | production | 99.00% | 0.9900 | 0.9898 | 0.9898 | 99.94% | 4.26M | 16.4 MB | 13.17 | 0.0500 |
| **★** | Custom CNN + CBAM | production | 98.42% | 0.9845 | 0.9842 | 0.9842 | 99.90% | 1.26M | 4.8 MB | 7.41 | 0.0245 |
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
| Max softmax probability | 0.9752 | 0.8059 | 93.4% |
| Normalised entropy | 0.9770 | 0.2700 | 93.2% |
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
