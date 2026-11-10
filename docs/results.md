<!-- GENERATED FILE - do not edit by hand.
     Produced by scripts/generate_docs.py from artifacts/results/experiments.json.
     Re-run that script after any training sweep. -->

# Results

_Generated 2026-11-10 14:37 UTC from `artifacts/results/experiments.json` (36 completed experiments)._

## Headline

| Result | Model | Protocol | Value |
|---|---|---|---|
| Best accuracy | ResNet50 + CBAM | field_long_resnet50 | **99.74%** |
| Best macro F1 | ResNet50 + CBAM | field_long_resnet50 | **0.9973** |
| Fastest inference | Logistic Regression (hand-crafted features) | classical | 0.02 ms/image |
| Smallest model | Ablation A: CNN | ablation | 4.8 MB |
| Best proposed (CBAM) model | EfficientNet-B0 + CBAM | field_long | 99.68% accuracy, 0.9967 macro F1 |
| Serving in the application | EfficientNet-B0 + CBAM & ResNet50 + CBAM (averaged) | field_real_max_ensemble | selected by `f1_macro` |
| Evaluated on | held-out test split | — | 17,572 images, 38 classes |

Rows with different protocols are **not** comparable: `benchmark` models were trained on a fixed subset and scored on a 4,560-image split; `production`, `field` and `field_long` models were trained on the full data and scored on 17,572 images.

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


## The deployed model: EfficientNet-B0 + CBAM & ResNet50 + CBAM (averaged)

Every metric for the model the application serves (`ensemble_cbam__fieldrealmaxens__1abec594`), on the full held-out test split.

| Metric | Value |
|---|---:|
| Test images | 17,572 |
| Accuracy (top-1) | 99.74% |
| Top-3 accuracy | 99.99% |
| Top-5 accuracy | 99.99% |
| Balanced accuracy | 99.73% |
| Macro precision | 0.9974 |
| Macro recall | 0.9973 |
| Macro F1 | 0.9973 |
| Weighted F1 | 0.9974 |
| Cohen's kappa | 0.9973 |
| Matthews correlation | 0.9973 |
| ROC AUC (macro, one-vs-rest) | 0.99996 |
| Expected calibration error (raw) | 0.0565 |
| Brier score | 0.0079 |
| Mean confidence when right / wrong | 94.2% / 63.2% |
| Inference, batch 1 | 37.11 ms (cuda) |

46 of 17,572 test images are misclassified.


## Per-class accuracy (deployed model)

Sorted hardest first. Recall is the per-class accuracy: the share of that disease's test images the model gets right.

| # | Class | Recall (accuracy) | Precision | F1 | Test images |
|---:|---|---:|---:|---:|---:|
| 1 | Corn (maize) - Cercospora leaf spot Gray leaf spot | 96.34% | 99.00% | 0.9765 | 410 |
| 2 | Corn (maize) - Northern Leaf Blight | 99.37% | 96.73% | 0.9804 | 477 |
| 3 | Tomato - Target Spot | 97.16% | 100.00% | 0.9856 | 457 |
| 4 | Tomato - Septoria leaf spot | 99.77% | 98.42% | 0.9909 | 436 |
| 5 | Tomato - Bacterial spot | 100.00% | 98.38% | 0.9918 | 425 |
| 6 | Tomato - Early blight | 99.17% | 99.37% | 0.9927 | 480 |
| 7 | Tomato - Spider mites Two-spotted spider mite | 99.54% | 99.31% | 0.9943 | 435 |
| 8 | Tomato - Late blight | 99.57% | 99.57% | 0.9957 | 463 |
| 9 | Tomato - Tomato Yellow Leaf Curl Virus | 99.39% | 99.80% | 0.9959 | 490 |
| 10 | Tomato - Healthy | 100.00% | 99.59% | 0.9979 | 481 |
| 11 | Strawberry - Leaf scorch | 99.77% | 100.00% | 0.9989 | 444 |
| 12 | Corn (maize) - Common rust | 99.79% | 100.00% | 0.9990 | 477 |
| 13 | Apple - Healthy | 100.00% | 99.80% | 0.9990 | 502 |
| 14 | Apple - Apple scab | 99.80% | 100.00% | 0.9990 | 504 |
| 15 | Apple - Black rot | 100.00% | 100.00% | 1.0000 | 497 |
| 16 | Apple - Cedar apple rust | 100.00% | 100.00% | 1.0000 | 440 |
| 17 | Blueberry - Healthy | 100.00% | 100.00% | 1.0000 | 454 |
| 18 | Cherry (including sour) - Powdery mildew | 100.00% | 100.00% | 1.0000 | 421 |
| 19 | Cherry (including sour) - Healthy | 100.00% | 100.00% | 1.0000 | 456 |
| 20 | Corn (maize) - Healthy | 100.00% | 100.00% | 1.0000 | 465 |
| 21 | Grape - Black rot | 100.00% | 100.00% | 1.0000 | 472 |
| 22 | Grape - Esca (Black Measles) | 100.00% | 100.00% | 1.0000 | 480 |
| 23 | Grape - Leaf blight (Isariopsis Leaf Spot) | 100.00% | 100.00% | 1.0000 | 430 |
| 24 | Grape - Healthy | 100.00% | 100.00% | 1.0000 | 423 |
| 25 | Orange - Haunglongbing (Citrus greening) | 100.00% | 100.00% | 1.0000 | 503 |
| 26 | Peach - Bacterial spot | 100.00% | 100.00% | 1.0000 | 459 |
| 27 | Peach - Healthy | 100.00% | 100.00% | 1.0000 | 432 |
| 28 | Pepper, bell - Bacterial spot | 100.00% | 100.00% | 1.0000 | 478 |
| 29 | Pepper, bell - Healthy | 100.00% | 100.00% | 1.0000 | 497 |
| 30 | Potato - Early blight | 100.00% | 100.00% | 1.0000 | 485 |
| 31 | Potato - Late blight | 100.00% | 100.00% | 1.0000 | 485 |
| 32 | Potato - Healthy | 100.00% | 100.00% | 1.0000 | 456 |
| 33 | Raspberry - Healthy | 100.00% | 100.00% | 1.0000 | 445 |
| 34 | Soybean - Healthy | 100.00% | 100.00% | 1.0000 | 505 |
| 35 | Squash - Powdery mildew | 100.00% | 100.00% | 1.0000 | 434 |
| 36 | Strawberry - Healthy | 100.00% | 100.00% | 1.0000 | 456 |
| 37 | Tomato - Leaf Mold | 100.00% | 100.00% | 1.0000 | 470 |
| 38 | Tomato - Tomato mosaic virus | 100.00% | 100.00% | 1.0000 | 448 |

27 of 38 classes are recognised with 100% recall; the hardest is **Corn (maize) - Cercospora leaf spot Gray leaf spot** at 96.34%.

## Benchmark

Every model in the `benchmark` protocol was trained on an identical fixed-budget subset (300 train / 60 val / 120 test images per class, 160 px, 12 epochs, identical optimiser and seed). Only the architecture differs. Full-data runs (`production` and `field`) are in their own section below and are **not** directly comparable with these subset rows.

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


## Full-data runs

Trained on the complete training split at 224 px and scored on the full 17,572-image test set, so these rows are comparable with each other. The `production` protocol uses standard augmentation; `field` and `field_long` add field-condition augmentation (see [Field-condition training](#field-condition-training)).

| | Model | Protocol | Accuracy | Precision | Recall | Macro F1 | Top-3 | Params | Size | ms/img | ECE |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|  | ResNet50 + CBAM | field_long_resnet50 | 99.74% | 0.9973 | 0.9973 | 0.9973 | 99.98% | 24.11M | 92.2 MB | 11.01 | 0.0472 |
| **★** | EfficientNet-B0 + CBAM | field_long | 99.68% | 0.9968 | 0.9967 | 0.9967 | 99.98% | 4.26M | 16.4 MB | 18.82 | 0.0437 |
|  | ResNet50 + CBAM | field_real_max_resnet50 | 99.67% | 0.9967 | 0.9966 | 0.9966 | 99.98% | 24.11M | 92.2 MB | 9.18 | 0.0462 |
| **★** | EfficientNet-B0 + CBAM | field_real_max | 99.60% | 0.9960 | 0.9959 | 0.9959 | 99.98% | 4.26M | 16.4 MB | 15.12 | 0.0536 |
| **★** | EfficientNet-B0 + CBAM | field_real | 99.56% | 0.9956 | 0.9954 | 0.9955 | 99.98% | 4.26M | 16.4 MB | 14.56 | 0.0527 |
| **★** | EfficientNet-B0 + CBAM | field_real_all | 99.53% | 0.9952 | 0.9952 | 0.9952 | 99.98% | 4.26M | 16.4 MB | 13.35 | 0.0506 |
| **★** | EfficientNet-B0 + CBAM | field_real_plus | 99.33% | 0.9933 | 0.9931 | 0.9931 | 99.98% | 4.26M | 16.4 MB | 14.02 | 0.0470 |
|  | ResNet50 + CBAM | field_real_plus_resnet50 | 99.23% | 0.9922 | 0.9922 | 0.9922 | 99.97% | 24.11M | 92.2 MB | 9.10 | 0.0458 |
|  | EfficientNet-B0 | production | 99.06% | 0.9906 | 0.9904 | 0.9904 | 99.94% | 4.06M | 15.6 MB | 18.50 | 0.0500 |
| **★** | EfficientNet-B0 + CBAM | production | 99.00% | 0.9900 | 0.9898 | 0.9898 | 99.94% | 4.26M | 16.4 MB | 13.17 | 0.0500 |
|  | CBAM on last two stages | production | 98.81% | 0.9881 | 0.9881 | 0.9880 | 99.94% | 1.26M | 4.8 MB | 4.18 | 0.0275 |
| **★** | Custom CNN + CBAM | production | 98.42% | 0.9845 | 0.9842 | 0.9842 | 99.90% | 1.26M | 4.8 MB | 7.41 | 0.0245 |
|  | EfficientNet-B0 | field | 98.32% | 0.9833 | 0.9829 | 0.9828 | 99.87% | 4.06M | 15.6 MB | 18.07 | 0.0501 |
| **★** | EfficientNet-B0 + CBAM | field | 98.25% | 0.9828 | 0.9822 | 0.9821 | 99.87% | 4.26M | 16.4 MB | 13.92 | 0.0497 |


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
| EfficientNet-B0 + CBAM | 0.0437 | 0.0017 | 0.536 | +96% |
| ResNet50 + CBAM | 0.0458 | 0.0467 | 1.003 | -2% |
| ResNet50 + CBAM | 0.0462 | 0.0558 | 1.031 | -21% |
| EfficientNet-B0 + CBAM | 0.0470 | 0.0469 | 1.000 | +0% |
| CBAM on every stage | 0.0472 | 0.0057 | 0.670 | +88% |
| ResNet50 + CBAM | 0.0472 | 0.0013 | 0.605 | +97% |
| EfficientNet-B0 + CBAM | 0.0497 | 0.0027 | 0.588 | +95% |
| EfficientNet-B0 + CBAM | 0.0500 | 0.0023 | 0.520 | +95% |
| EfficientNet-B0 | 0.0500 | 0.0022 | 0.507 | +96% |
| EfficientNet-B0 | 0.0501 | 0.0021 | 0.577 | +96% |
| EfficientNet-B0 + CBAM | 0.0506 | 0.0358 | 0.941 | +29% |
| EfficientNet-B0 + CBAM | 0.0527 | 0.0452 | 0.973 | +14% |
| EfficientNet-B0 + CBAM | 0.0536 | 0.0532 | 0.999 | +1% |
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
| Max softmax probability | 0.9796 | 0.8990 | 94.2% |
| Normalised entropy | 0.9799 | 0.1872 | 94.0% |
| Free energy | 0.5185 | 0.0000 | 1.7% |


### Whole-pipeline rejection

The image-quality gate runs *before* the model, so the end-to-end rate is what a user actually experiences.

| Stage | OOD rejected |
|---|---:|
| Image-quality gate alone | 66.8% |
| OOD score alone | 94.3% |
| **Combined pipeline** | **97.2%** |
| Cost: genuine leaves rejected | 3.0% |


| OOD family | Quality gate | OOD score | Combined |
|---|---:|---:|---:|
| extreme | 64.0% | 85.0% | 93.0% |
| flat | 100.0% | 100.0% | 100.0% |
| noise | 100.0% | 100.0% | 100.0% |
| shuffle | 3.0% | 95.0% | 96.0% |


> **Caveat.** Synthetic proxies (uniform noise, flat colour fields, tile-shuffled leaves, heavily degraded leaves) are genuinely out of distribution but are easier to reject than real photographs of unrelated subjects. The detection rates below are therefore an upper bound. Re-run with --ood-dir pointing at a real non-leaf corpus for a tighter estimate.


## Classical baselines

Trained on 1184 hand-crafted features (colour histograms and moments, LBP, GLCM, HOG) extracted from 11,400 images.


**Not trained:** RBF SVM. 
The RBF SVM is skipped by default: `probability=True` fits five internal cross-validation folds of an O(n^2) solver over 11,400 samples, which costs hours on this hardware for a baseline the linear SVM already covers. Run it explicitly with `python training/train_classical.py --models rbf_svm` if the comparison is wanted.


## Field-condition training

Every score above comes from PlantVillage's laboratory photographs, and the first measurement through the full pipeline showed the gap: the model trained on lab photos fell to about 70% when the same leaves were given a soil backdrop, harsh sun and heavy JPEG together, and to under 10% when the leaf was small in the frame. Two training rounds added **field-condition augmentation** (`backend/app/ml/field_augment.py`: leaf cut-outs on soil, mulch, foliage and stone backdrops, small-in-frame placement, directional light, blur, noise, JPEG - the leaf's hue is never altered).

| Model | Protocol | Epochs run | Augmentation | Test accuracy | Macro F1 | Test errors | Serving |
|---|---|---:|---|---:|---:|---:|:---:|
| EfficientNet-B0 | production | 6/6 | medium | 99.06% | 0.9904 | 166 |  |
| EfficientNet-B0 + CBAM | production | 6/6 | medium | 99.00% | 0.9898 | 176 |  |
| EfficientNet-B0 | field | 6/6 | field | 98.32% | 0.9828 | 296 |  |
| EfficientNet-B0 + CBAM | field | 6/6 | field | 98.25% | 0.9821 | 307 |  |
| EfficientNet-B0 + CBAM | field_long | 9/10 † | field_mild | 99.68% | 0.9967 | 56 |  |
| ResNet50 + CBAM | field_long_resnet50 | 10/10 | field_mild | 99.74% | 0.9973 | 46 |  |
| EfficientNet-B0 + CBAM | field_real | 6/6 | field_mild | 99.56% | 0.9955 | 78 |  |
| EfficientNet-B0 + CBAM | field_real_all | 6/6 | field_mild | 99.53% | 0.9952 | 83 |  |
| EfficientNet-B0 + CBAM | field_real_plus | 10/10 | field_mild | 99.33% | 0.9931 | 118 |  |
| EfficientNet-B0 + CBAM | field_real_max | 10/10 | field_mild | 99.60% | 0.9959 | 70 |  |
| ResNet50 + CBAM | field_real_plus_resnet50 | 10/10 | field_mild | 99.23% | 0.9922 | 136 |  |
| ResNet50 + CBAM | field_real_max_resnet50 | 10/10 | field_mild | 99.67% | 0.9966 | 58 |  |

`production`: standard augmentation, last 3 blocks fine-tuned, 6 epochs. `field`: field augmentation on 70% of images, otherwise the same schedule. `field_long`: field augmentation on 50% of images, **the whole network fine-tuned**, 10 epochs. All scored on the same 17,572-image test split.

† The `field_long` run was interrupted by a machine restart during epoch 10 of 10. The trainer saves the best-so-far checkpoint whenever validation improves and tests that file at the end, so the epoch-9 checkpoint on disk is the model a finished run would have been tested on (unless the unfinished epoch improved on it, at a learning rate near zero). `scripts/recover_interrupted_run.py` evaluated it exactly as a normal run is, and the ledger record is flagged `interrupted`. Its history is reconstructed from the training log to four decimals.

### Behaviour on field-like photos

The same 26 disease classes x 4 held-out leaves, put through each field condition once and scored by every model identically - through the full serving pipeline (quality gate, dust check, classifier, OOD check) with one set of OOD thresholds for all, so a difference is the network. Cells show the share classified correctly, with the number of confident wrong answers (status `ok` but wrong) in brackets.

| Model | Lab test | Original lab photo | Leaf on a soil background | Photographed off-axis | Leaf small in the frame (far away) | Harsh directional sunlight | Heavy JPEG compression | All of the above (closest to a field photo) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EfficientNet-B0 [production] | 99.06% | 96.2% (1) | 96.2% (0) | 96.2% (1) | 6.7% (1) | 97.1% (1) | 92.3% (1) | 69.2% (1) |
| EfficientNet-B0 + CBAM [production] | 99.00% | 96.2% (1) | 96.2% (0) | 96.2% (1) | 8.7% (0) | 97.1% (1) | 92.3% (1) | 71.2% (1) |
| EfficientNet-B0 [field] | 98.32% | 95.2% (1) | 97.1% (0) | 95.2% (1) | 64.4% (7) | 96.2% (1) | 92.3% (0) | 85.6% (1) |
| EfficientNet-B0 + CBAM [field] | 98.25% | 95.2% (1) | 95.2% (0) | 96.2% (1) | 60.6% (10) | 95.2% (1) | 91.3% (0) | 87.5% (2) |
| EfficientNet-B0 + CBAM [field_long] | 99.68% | 96.2% (1) | 99.0% (0) | 97.1% (1) | 94.2% (0) | 97.1% (1) | 96.2% (0) | 98.1% (1) |
| ResNet50 + CBAM [field_long_resnet50] | 99.74% | 97.1% (0) | 99.0% (0) | 98.1% (0) | 94.2% (1) | 96.2% (0) | 95.2% (0) | 97.1% (2) |
| EfficientNet-B0 + CBAM [field_real_plus] | 99.33% | 97.1% (0) | 99.0% (0) | 97.1% (0) | 91.3% (3) | 98.1% (0) | 94.2% (2) | 92.3% (2) |
| EfficientNet-B0 + CBAM [field_real_max] | 99.60% | 96.2% (0) | 99.0% (0) | 98.1% (0) | 90.4% (2) | 97.1% (0) | 95.2% (1) | 94.2% (1) |
| EfficientNet-B0 + CBAM & ResNet50 + CBAM (averaged) [field_real_max_ensemble] **←** | 99.74% | 97.1% (0) | 100.0% (0) | 98.1% (0) | 93.3% (0) | 98.1% (0) | 96.2% (0) | 97.1% (0) |
| ResNet50 + CBAM [field_real_plus_resnet50] | 99.23% | 95.2% (0) | 98.1% (1) | 95.2% (1) | 85.6% (7) | 97.1% (1) | 94.2% (2) | 94.2% (3) |
| ResNet50 + CBAM [field_real_max_resnet50] | 99.67% | 97.1% (0) | 100.0% (0) | 98.1% (0) | 93.3% (3) | 98.1% (0) | 97.1% (0) | 98.1% (0) |

n = 104 images per condition. A refusal produces no class and counts as not correct. The row marked **←** is the model the application serves.

### Why this model serves

The average of the two networks fine-tuned on the eight real-photo sources (EfficientNet-B0 + CBAM and ResNet50 + CBAM, each with its own fitted temperature). Chosen on validation photographs, never the test splits: of the configurations the free host can serve (at most two networks, no test-time flipping), it has the highest PlantVillage validation accuracy among those at least as accurate as the single EfficientNet on the real web-photo validation sets. On the test splits it is better than the single EfficientNet on both: lab accuracy 99.74% (was 99.60%) and real photos 73.7% top-1, 93.2% top-3 (was 71.6%, 91.1%).

See [Accuracy on real photographs](#accuracy-on-real-photographs).

### Reading these results honestly

* **Not independent of the training.** The augmentation imitates the same kinds of condition the field-like test applies (through different generators), so the field-like scores flatter the field-trained models. Real field photographs are the independent test - see [Accuracy on real photographs](#accuracy-on-real-photographs).
* **The gain cannot be attributed to CBAM.** Round one trained plain EfficientNet-B0 and EfficientNet-B0 + CBAM identically and they tied (98.32% vs 98.25%). Round two trained only the CBAM model, with a longer schedule, the whole network unfrozen and milder augmentation changed together, so it can't separate the contribution of any one of those changes either - and no plain-EfficientNet control was run on that schedule. The earlier `production` models used a shorter, shallower schedule (6 epochs, last 3 blocks fine-tuned), so some of the gain may simply be that the original models were not trained as far as they could be.
* **Round one cost lab accuracy** (about 0.7 points); the milder, longer round two more than recovered it.


## Accuracy on real photographs

Every other table uses PlantVillage laboratory photographs, or lab photographs altered to look like field shots. This one uses **real photographs taken in fields or collected from the web**: the held-out test split of PlantDoc (Singh et al., CODS-COMAD 2020; CC-BY-4.0), 236 photos. Its classes map onto the model's own, so nothing about the model's outputs changes.

Before any training, 16 PlantDoc training photos were removed because a perceptual hash matched a test photo (10 of them carried a different label), so no test photo was seen in training.

| Model | Trained on | Real photos: top-1 (95% CI) | Top-3 | Shown right | Shown **wrong** | Lab test accuracy |
|---|---|---:|---:|---:|---:|---:|
| EfficientNet-B0 | lab photos only | 30.1% (25%–36%) | 52.1% | 11.4% | 8.9% | 99.06% |
| EfficientNet-B0 + CBAM | lab photos only | 31.8% (26%–38%) | 59.7% | 16.9% | 14.0% | 99.68% |
| ResNet50 + CBAM | lab photos only | 37.7% (32%–44%) | 66.5% | 25.0% | 21.2% | 99.74% |
| EfficientNet-B0 + CBAM | lab + 1 real-photo dataset, 6 epochs | 66.5% (60%–72%) | 90.7% | 46.2% | 11.9% | 99.56% |
| EfficientNet-B0 + CBAM | lab + 3 real-photo datasets, 6 epochs | 67.4% (61%–73%) | 90.7% | 48.7% | 13.1% | 99.53% |
| EfficientNet-B0 + CBAM | lab + 5 real-photo datasets, 10 epochs | 72.0% (66%–77%) | 92.8% | 54.2% | 8.9% | 99.33% |
| ResNet50 + CBAM | lab + 5 real-photo datasets, 10 epochs | 72.5% (66%–78%) | 89.8% | 61.4% | 15.7% | 99.23% |
| EfficientNet-B0 + CBAM | lab (800/class) + 8 real-photo datasets, 10 epochs | 71.6% (66%–77%) | 91.1% | 55.5% | 14.0% | 99.60% |
| ResNet50 + CBAM | lab (800/class) + 8 real-photo datasets, 10 epochs | 73.7% (68%–79%) | 89.8% | 59.7% | 14.4% | 99.67% |
| EfficientNet-B0 + CBAM & ResNet50 + CBAM (averaged) **←** | average of the two models above | 73.7% (68%–79%) | 93.2% | 54.2% | 9.7% | 99.74% |

*Top-1/top-3*: the network alone, on every photo. *Shown right / wrong*: what the full application displays with status `ok` (the rest are flagged as uncertain, dusty or not identified, or refused), with every model judged at the one "not sure" threshold now in force - so these columns compare networks, not the settings each was originally served with. The row marked **←** is the model the application serves.

**What this means.** Real photographs are much harder than laboratory ones: several leaves, real backgrounds, different lighting, cultivars and disease stages. A model's PlantVillage accuracy (~99.7%) is not its accuracy on a photo from a phone or a web search, and this table is the closer estimate. PlantDoc's labels are themselves known to contain some noise, and its test split is small - read differences smaller than the intervals as ties.

### Two networks, averaged - the served model

EfficientNet-B0 + CBAM and ResNet50 + CBAM (24 M parameters, 92 MB) were fine-tuned on the same eight sources with the same schedule. The application serves the average of their calibrated probabilities (`ensemble_cbam`, assembled by `scripts/build_ensemble.py`). Top-1 / top-3 on real photos:

| Real-photo split | EfficientNet-B0 + CBAM | ResNet50 + CBAM | Average of both |
|---|---:|---:|---:|
| plantdoc_test (236) | 71.6% / 91.1% | 73.7% / 89.8% | 73.7% / 93.2% |
| plantdoc_val (348) | 75.3% / 89.7% | 73.0% / 88.5% | 74.1% / 91.7% |
| plantwild_val (734) | 69.3% / 86.6% | 71.0% / 87.7% | 72.8% / 88.8% |
| plantpathology2021_val (346) | 96.8% / 99.4% | 97.4% / 99.4% | 98.0% / 99.7% |
| ccmt_val (347) | 90.8% / 98.3% | 90.2% / 99.1% | 91.9% / 99.4% |

Alone, the two networks are within about two points of each other; averaged, they match or beat the better one on every real-photo set but one and are clearly ahead at top-3 - and on the PlantVillage test split they make 46 mistakes in 17,572 (99.74%) against 70 and 58 for the two alone. That puts the system at the top of the 60-76% range published for PlantDoc. The pair was chosen on validation photographs, never the test splits: of the configurations the free host can serve (at most two networks, no test-time flipping), the one with the highest PlantVillage validation accuracy among those at least as accurate as the single EfficientNet on the real web-photo validation sets. Larger combinations (three or four networks, with mirrored copies) scored higher on validation but reached at most 99.77% on the test split, not 99.8%. Serving the pair costs about 206 MB of memory (one network: 108 MB) and about 94 ms of CPU per photo on one thread (16 ms), within the free host's 512 MB.

### Photos from the newest field sources

Two of the sources added last are field collections the earlier models never saw. Top-1 / top-3 on their held-out test splits - same-source tests, so easier than PlantDoc's, but they show what the extra data buys:

| Held-out photos | EfficientNet-B0 + CBAM & ResNet50 + CBAM (averaged) (served) | EfficientNet-B0 + CBAM (previous) |
|---|---:|---:|
| Vineyard photos (GVLiD test split) (447) | 98.9% / 99.8% | 32.0% / 47.4% |
| Corn-field photos (CD&S test split) (204) | 99.0% / 100.0% | 58.8% / 98.5% |

### Real-photo training data

Fine-tuning used a stratified PlantVillage subset (so no class is forgotten) plus these real-photo sources. Only classes that map unambiguously onto the model's 38 were used; photos were resized to 512 px, de-duplicated, and any photo matching a PlantDoc test photo was removed.

| Source | Photos used | Licence | Citation |
|---|---:|---|---|
| [plantdoc](https://github.com/pratikkayal/PlantDoc-Dataset) | 2,562 | CC-BY-4.0 | Singh et al., PlantDoc: A Dataset for Visual Plant Disease Detection, CODS-COMAD 2020 |
| [plantpathology2021](https://huggingface.co/datasets/timm/plant-pathology-2021) | 3,464 | Apache-2.0 (Hugging Face mirror) | Thapa et al., Plant Pathology 2021 - FGVC8 (Cornell) |
| [ccmt](https://data.mendeley.com/datasets/bwh3zbpkpv/1) | 3,468 | CC BY 4.0 | Mensah Kwabena et al., Dataset for Crop Pest and Disease Detection, Mendeley Data v1, 2023 (doi:10.17632/bwh3zbpkpv.1) |
| [plantwild](https://huggingface.co/datasets/uqtwei2/PlantWild) | 7,316 | CC BY-NC-ND 4.0 (non-commercial) | Wei et al., Benchmarking In-the-Wild Multimodal Plant Disease Recognition and A Versatile Baseline, ACM MM 2024 |
| [potato_field](https://data.mendeley.com/datasets/ptz377bwb8/1) | 509 | CC BY 4.0 | Shabrina et al., A novel dataset of potato leaf disease in uncontrolled environment, Data in Brief 2024 (doi:10.17632/ptz377bwb8.1) |
| [plantseg](https://zenodo.org/records/17719108) | 121 | CC BY-NC 4.0 (non-commercial) | Wei et al., PlantSeg: A Large-Scale In-the-wild Dataset for Plant Disease Segmentation (Zenodo v7, 2025) |
| [gvlid](https://data.mendeley.com/datasets/wkymf8bhcg/5) | 2,238 | CC BY 4.0 | GVLiD: GrapeVine Leaf identification of the Diseases, Mendeley Data v5 (doi:10.17632/wkymf8bhcg.5) |
| [cds](https://osf.io/s6ru5/) | 1,020 | no licence stated (used for non-commercial academic work, with citation) | Ahmad et al., CD&S Dataset: Handheld Imagery Dataset Acquired Under Field Conditions for Corn Disease Identification and Severity Estimation, 2021 |

PlantWild (CC BY-NC-ND 4.0) and PlantSeg (CC BY-NC 4.0) are licensed for non-commercial use only, and CD&S states no licence: a model fine-tuned on them should be used for research and education, which is this project's purpose. No photos from any source are committed to this repository.

### When the app says "not sure"

Below a minimum top-class probability the app does not give a plain diagnosis; it says it is unsure and shows the closest matches. That threshold was chosen on 2,095 real validation photographs (`scripts/choose_real_operating_point.py`; never the PlantDoc test split) as the most permissive one at which answers shown as a diagnosis are right at least 90% of the time - and at least 85% on the 1,045 photos found through web image searches (PlantDoc, PlantWild, PlantSeg) on their own. The field collections are easy, and pooled with them a threshold can look precise overall while letting through confidently wrong answers on exactly the photos people check against a web search; no threshold reaches 90% on the web photos alone, so their floor is 85%, slightly above what the previous model achieved there (84.7%).

| Threshold | Shown as diagnosis | Of those, right | Confidently wrong (of all photos) | Web photos shown | Of those, right |
|---:|---:|---:|---:|---:|---:|
| 0.10 | 100.0% | 84.5% | 15.5% | 100.0% | 72.9% |
| 0.20 | 99.7% | 84.6% | 15.3% | 99.5% | 73.0% |
| 0.30 | 98.2% | 85.5% | 14.3% | 96.5% | 74.3% |
| 0.40 | 95.8% | 87.1% | 12.4% | 92.0% | 77.1% |
| 0.50 | 90.1% | 90.0% | 9.0% | 81.7% | 81.9% |
| 0.60 | 85.0% | 91.5% | 7.2% | 73.3% | 84.1% |
| 0.69 **(in force)** | 80.8% | 92.3% | 6.3% | 67.0% | 85.3% |
| 0.70 | 80.4% | 92.4% | 6.1% | 66.3% | 85.4% |
| 0.80 | 75.6% | 93.9% | 4.6% | 59.4% | 87.9% |
| 0.90 | 65.8% | 95.2% | 3.2% | 46.5% | 89.5% |

At the previous setting (0.15) this model would show every real photo as a diagnosis, 15.4% of them confidently wrong; at 0.69 that falls to 6.3%, at the cost of saying "not sure" for 19.2% of photos - for which the right answer is usually among the closest matches shown.


## Accuracy on field-like photos

Every number above comes from laboratory photographs. This measures how the full pipeline (quality gate, dust check, model, OOD check) behaves when the same leaves are made to look like field photos - 104 disease images per condition, produced by `scripts/domain_shift_probe.py`.

| Condition | Correct class | vs lab | Not marked reliable |
|---|---:|---:|---:|
| Original lab photo | 93.3% (97/104) | — | 10.6% |
| Leaf on a soil background | 98.1% (102/104) | +4.8 pt | 6.7% |
| Photographed off-axis | 98.1% (102/104) | +4.8 pt | 9.6% |
| Leaf small in the frame (far away) | 84.6% (88/104) | -8.7 pt | 26.9% |
| Harsh directional sunlight | 96.2% (100/104) | +2.9 pt | 4.8% |
| Heavy JPEG compression | 94.2% (98/104) | +1.0 pt | 15.4% |
| All of the above (closest to a field photo) | 93.3% (97/104) | +0.0 pt | 30.8% |

*Correct class* counts the top prediction when the app gave one, including a provisional or low-confidence answer. A refusal (poor quality, not a plant, heavy dust) produces no class and counts as not correct - which is why even the unmodified photos score below the model's lab test accuracy: the pipeline's own checks refuse a few percent of clean laboratory photos. *Not marked reliable* is any outcome other than a plain `ok`.

The hardest single condition is **leaf small in the frame (far away)** (84.6%). When the leaf is small in the frame the app advises moving closer.

> Transformations imitate field conditions applied to laboratory images. Real photographs additionally bring occlusion, motion blur, overlapping leaves and unseen species, so this is a lower bound on the accuracy drop rather than an estimate of it.


## Dust, soil and sand detection

PlantVillage has no "dusty leaf" class, so a dust film used to be classified as a disease (often powdery mildew) or could hide one. A small detector in `backend/app/ml/dust.py` flags it before diagnosis. Evaluated on 4,770 held-out images (AUC **0.972**; 0.935 on crops withheld from training: Squash, Cherry_(including_sour), Corn_(maize), Grape).

| Image | n | Flagged "possible dust" (provisional) | Flagged "heavy dust" (withheld) |
|---|---:|---:|---:|
| Clean leaf (lab photo) | 760 | 1.2% | 0.1% |
| Clean leaf, field-like conditions | 760 | 5.0% | 0.9% |
| Light dust | 211 | 67.3% | 32.2% |
| Moderate dust | 250 | 86.8% | 59.6% |
| Heavy dust | 299 | 97.3% | 78.3% |
| Light dust, field-like | 189 | 45.5% | 13.2% |
| Moderate dust, field-like | 243 | 67.1% | 29.6% |
| Heavy dust, field-like | 328 | 80.5% | 49.7% |

Rows marked *clean* are false alarms. Genuine powdery-mildew leaves (n=99) trip the possible-dust warning 10.1% of the time and the heavy-dust refusal 3.0%, because a pale fungal coating and dust look alike; that is why the warning says so instead of refusing.

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

Measured through the running API with `scripts/benchmark_serving.py` on 2026-11-01: EfficientNet-B0 + CBAM on cuda, 40 requests, with the dust/soil check included in the quality stage. Re-run the script after changing the model or the gate.

| Stage | Median | p95 |
|---|---:|---:|
| Image-quality checks | 13.7 ms | 18.2 ms |
| Preprocessing | 1.5 ms | 2.1 ms |
| Model inference | 20.0 ms | 24.9 ms |
| Explanation heat maps | 238.4 ms | 369.7 ms |
| **Total request** | 277.8 ms | 404.2 ms |

Throughput: 3.51 requests/s with explanations on. The free Render deployment runs the same model through ONNX Runtime on a shared CPU, so expect slower responses there.


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
