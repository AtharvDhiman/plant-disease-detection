# System architecture

## Overview

The project is four separable pieces with narrow interfaces between them. The
important boundary is between **training** and **serving**: they share model
code but nothing else, and they communicate only through files on disk.

```mermaid
flowchart LR
    subgraph Offline["Offline — training pipeline"]
        DS[(Kaggle dataset)]
        INS[dataset_inspect.py]
        SPL[prepare_splits.py]
        TRN[run_experiments.py]
        EVL[evaluate.py]
        EXP[export_model.py]
        DS --> INS --> SPL --> TRN --> EVL --> EXP
    end

    subgraph Artifacts["Artefacts on disk — the only interface"]
        CKPT[/models/checkpoints/*.pt/]
        MAN[/models/exported/production.json/]
        OOD[/models/metadata/ood_thresholds.json/]
        LEDGER[/artifacts/results/experiments.json/]
        KB[/data/disease_info.json/]
    end

    EXP --> MAN
    TRN --> CKPT
    TRN --> LEDGER

    subgraph Online["Online — FastAPI service"]
        REG[ModelRegistry<br/>loads once at startup]
        PRED[Predictor]
        API[REST API]
        DB[(SQLite)]
        MAN --> REG --> PRED --> API
        OOD --> PRED
        KB --> PRED
        LEDGER --> API
        API <--> DB
    end

    subgraph Client["Browser — React SPA"]
        UI["Upload and result"]
        DASH[Dashboard]
        BENCH[Benchmark]
        RES[Research]
    end

    API --> UI
    API --> DASH
    API --> BENCH
    API --> RES
```

**Why this boundary matters.** The FastAPI process cannot train, cannot read the
dataset, and cannot compute a metric. It reads a checkpoint and some JSON. That
means a model can be retrained and promoted without touching application code,
and it means no reported number can drift from the experiment that produced it.

## Request path for one prediction

```mermaid
sequenceDiagram
    participant U as Browser
    participant A as FastAPI
    participant Q as Quality checks
    participant M as Model (in memory)
    participant X as Explainability
    participant D as SQLite

    U->>A: POST /api/analyze-image
    A->>Q: sharpness, exposure, coherence, plant colour
    Q-->>U: quality band + specific advice

    U->>A: POST /api/predict (file)
    A->>A: validate type, size, re-decode, strip EXIF
    A->>Q: quality report
    alt quality has an error-level issue
        A-->>U: status = poor_quality (no classification)
    else quality acceptable
        A->>M: eval transform → forward pass (no_grad)
        M-->>A: logits
        A->>A: logits / T → calibrated probabilities
        A->>A: OOD assessment (MSP, entropy, energy)
        A->>X: Grad-CAM, Grad-CAM++, CBAM gate
        X-->>A: heat-map overlays (PNG)
        A->>D: persist prediction + timings
        A-->>U: class, confidence, top-k, heat maps, disease info
    end
```

## Layer responsibilities

| Layer | Location | Responsibility | Must not |
|---|---|---|---|
| Model code | `backend/app/ml/` | Architectures, attention, transforms, explainability, OOD, quality, features | Touch HTTP or the database |
| Training | `training/` | Data prep, training loop, evaluation, calibration, export | Be imported by the API |
| Services | `backend/app/services/` | Predictor, knowledge base, artefact reading | Contain HTTP details |
| API | `backend/app/api/routes/` | Request validation, serialisation, status codes | Contain ML logic |
| Frontend | `frontend/src/` | Presentation, interaction | Compute metrics |

`backend/app/ml/` being shared by both training and serving is deliberate and
load-bearing: it is what guarantees an image is preprocessed at inference
exactly as it was during training. A duplicated transform is one of the most
common silent accuracy killers in deployed vision systems.

## The proposed model

```mermaid
flowchart TB
    IN[Input 224x224x3] --> PRE[Resize 255 → centre-crop 224<br/>ImageNet normalisation]
    PRE --> B1

    subgraph Stage["Convolutional stage x4"]
        B1[Conv 3x3 → BatchNorm → ReLU]
        B2[Conv 3x3 → BatchNorm → ReLU]
        CA[CBAM channel attention<br/>avg-pool + max-pool → shared MLP → sigmoid]
        SA[CBAM spatial attention<br/>channel avg + max → 7x7 conv → sigmoid]
        MP[MaxPool 2x2]
        B1 --> B2 --> CA --> SA --> MP
    end

    MP --> GAP[Global average pooling]
    GAP --> DO1[Dropout 0.4]
    DO1 --> FC1[Dense 256 → ReLU]
    FC1 --> DO2[Dropout 0.2]
    DO2 --> FC2[Dense 38]
    FC2 --> SM[Softmax]
    SM --> OUT[Class probabilities]
```

CBAM sits **after** the activated, batch-normalised feature map and **before**
down-sampling, which is where it has the most spatial detail to work with. The
`attention_stages` parameter controls which stages get a block; the placement
experiment measures whether all four, the last two, or only the last is best.

For transfer-learning variants the structure is the same idea applied once, at
the top of the pretrained trunk:

```
ImageNet backbone (frozen, then partially unfrozen)
  → [optional CBAM]
  → global average pooling
  → dropout
  → linear(num_classes)
```

## Data flow at training time

```mermaid
flowchart LR
    RAW[(train/ 70,295 images)] --> STRAT[Stratified 90/10 split]
    STRAT --> TR[train 63,265]
    STRAT --> VA[val 7,030]
    OFF[(valid/ 17,572 images)] --> TE[test 17,572<br/>never used for tuning]

    TR --> SUB1[Benchmark subset<br/>300/class]
    VA --> SUB2[60/class]
    TE --> SUB3[120/class]

    SUB1 --> BENCH[All 18 architectures<br/>identical protocol]
    TR --> PROD[Production run<br/>full data, 224px]
```

Manifests (`data/manifests/*.csv`) are materialised once so every script — the
classical baselines, the deep benchmark, the ablation — reads the same rows in
the same order. Nothing re-shuffles the data.

## Technology choices and their reasons

| Choice | Reason |
|---|---|
| PyTorch over TensorFlow | TensorFlow has no GPU support on Windows for Python 3.13 (native-Windows GPU builds stopped at TF 2.10 / Python 3.10). PyTorch CUDA works and was verified on the target GTX 1650. |
| FastAPI | Automatic OpenAPI, Pydantic validation at the boundary, async file uploads. |
| SQLite + SQLAlchemy | Zero-configuration for a single-node deployment; the ORM makes the swap to PostgreSQL a URL change. |
| Manifest CSVs over `ImageFolder` | Guarantees every experiment sees identical splits. |
| JSON ledger over MLflow | Runs from a clean clone with no server to start, and the API reads it directly. |
| React + Vite + Tailwind | Fast HMR, small bundle, no CSS naming overhead. |
| Recharts | Declarative, composes with React state, no imperative canvas layer. |

## Failure modes handled explicitly

| Failure | Handling |
|---|---|
| No model exported | API starts, `/api/predict` returns 503 with instructions; dashboard/docs still work |
| Corrupt image in a batch | Dataset substitutes the next readable sample and records the path; count reported after training |
| CUDA out of memory | Batch size halved and the run retried once |
| DataLoader workers cannot start | Detected and retried with in-process loading |
| Half precision slower than full | Measured at startup; AMP disabled when the measured speed-up is below 1.15x |
| Model file missing at load | Registry raises with the exact path; health endpoint reports it |
| Malformed experiment ledger | Moved aside and recreated rather than blocking a rerun |
