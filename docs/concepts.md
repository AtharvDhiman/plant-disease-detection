# Concepts: why each piece is in this project

Every design decision below is stated with its reason and, where relevant, with
the alternative that was rejected. This is the document to read before a viva.

---

## Why a CNN at all?

A leaf disease is identified by *local texture in a spatial arrangement*: a
brown lesion with a yellow halo, a powdery white film on the underside, a
rectangular streak bounded by veins. Three properties of convolution match that
problem exactly:

1. **Locality.** A lesion is a local pattern. A convolution kernel looks at a
   small neighbourhood, so it can learn "dark centre, light ring" without having
   to learn it separately for every position.
2. **Weight sharing.** The same lesion means the same thing wherever it appears
   on the leaf. A fully connected layer would need to learn the pattern
   independently at every pixel — orders of magnitude more parameters and far
   more data.
3. **Hierarchy.** Stacked layers compose edges → textures → lesion parts →
   lesion types. That hierarchy is what separates "yellowing" from "yellowing
   *between the veins*", which is the difference between two real diagnoses.

The classical-ML baselines in this project make the argument empirically rather
than by assertion: colour histograms, GLCM texture statistics and HOG are
exactly the hand-crafted features a practitioner would have used before deep
learning, and their measured scores are in the benchmark table next to the CNNs.

---

## Why attention, and why CBAM specifically?

A plain CNN treats every channel and every spatial position as equally worth
attending to. That is wasteful and, worse, it lets background pixels contribute
to the decision. Attention is a learned, input-dependent reweighting that fixes
both.

### Channel attention — *what* matters

A convolutional layer produces, say, 256 feature maps. For a given leaf, most of
them are irrelevant: a channel that fires on soil texture contributes nothing
when the leaf fills the frame. Channel attention pools each map to a single
number, passes those through a bottleneck MLP, and produces a gate per channel:

```
gate_c = σ( MLP(AvgPool(x)) + MLP(MaxPool(x)) )      x ← x ⊙ gate_c
```

The two pooling paths matter. **Average pooling** measures how *extensively* a
feature is present across the leaf; **max pooling** measures its *strongest*
evidence anywhere. A small but intense lesion has a low average and a high
maximum — average pooling alone would nearly miss it. The MLP is shared between
the two paths, so the gate is learned once and applied to both descriptors.

### Spatial attention — *where* it matters

Having decided which channels to trust, the network still needs to decide which
*pixels*. Spatial attention collapses the channel axis with average and max
pooling into a 2-channel map, then a single large-kernel (7×7) convolution
learns a per-pixel gate:

```
gate_s = σ( Conv7×7( [ mean_c(x) ; max_c(x) ] ) )     x ← x ⊙ gate_s
```

The large kernel is deliberate: deciding whether a region is lesion or
background needs surrounding context, not a 3×3 neighbourhood.

### Why CBAM rather than SE?

Squeeze-and-Excitation is channel attention only, and derives its descriptor
from average pooling alone. CBAM adds the max-pooling path *and* the spatial
branch. That is precisely the ablation this project runs — arms B (channel
only), C (spatial only), D (SE) and E (CBAM) against arm A (no attention) —
with everything else held constant, so the contribution of each component is
measured rather than argued.

### Why channel *then* spatial?

The original CBAM paper found sequential channel→spatial to beat both the
parallel arrangement and spatial→channel. The intuition: refining *what* to look
at first gives the spatial branch a cleaner signal to localise within.

### The honest caveat

CBAM adds parameters and latency. If the measured accuracy gain does not exceed
seed noise, the correct conclusion is that CBAM did not help *on this dataset at
this scale* — and that is what the ablation section reports. The dataset here is
laboratory imagery with uniform backgrounds, which is the regime where spatial
attention has the *least* to do; on cluttered field photographs the picture
could differ.

---

## Why transfer learning?

The benchmark subset gives each model 300 images per class. A network trained
from scratch must learn edge detectors, colour opponency and texture filters
from those 11,400 images. An ImageNet-pretrained backbone arrives already
knowing them, having seen 1.2 million images, and only needs to learn the
mapping from those generic features to 38 leaf classes.

### Why two stages?

**Stage 1 — frozen backbone.** The classification head starts random. Its first
gradients are large and badly directed. Letting them propagate into pretrained
weights on step one destroys the features that made the backbone worth using.
Freezing the trunk while the head finds a sensible region of parameter space
prevents that.

**Stage 2 — partial unfreeze at a lower learning rate.** Once the head is
reasonable, the upper backbone layers — the semantic ones — are unfrozen at 15%
of the base learning rate so they adapt to leaf pathology without being
overwritten. The lower layers stay frozen: edge and colour detectors are already
correct and re-learning them on 11,400 images would only overfit.

---

## Why Grad-CAM (and Grad-CAM++, and the attention map)?

A classifier that cannot be interrogated cannot be trusted in agriculture, where
a wrong answer costs a season's crop. Three complementary views are provided:

| Method | What it answers | Limitation |
|---|---|---|
| **Grad-CAM** | Which regions of the final feature map, if amplified, would most raise the score for the predicted class | Coarse — the final map is 7×7 or 14×14, upsampled |
| **Grad-CAM++** | Same, but weighted so several disjoint lesions all contribute | Same resolution limit |
| **CBAM spatial map** | Where the network *itself* chose to amplify features, read straight from the forward pass | Not class-specific; it is a property of the input |
| **Integrated gradients** | Pixel-level attribution satisfying a completeness axiom | Noisy; needs many forward passes |

**The caveat that matters, and that the UI states explicitly:** none of these is
a disease segmentation. They show where evidence was pooled from, at the
resolution of the feature map. A heat map covering a lesion is *evidence that
the model is looking at the right thing*; it is not a measurement of lesion
extent, and it must not be presented as one.

The practical value is diagnostic: if the heat map sits on the background rather
than the leaf, the model has latched onto a spurious correlation and the
prediction should be distrusted — regardless of how confident it is.

---

## Why a confidence threshold, and why calibrate it?

Softmax outputs look like probabilities but are not. A network trained with
cross-entropy is systematically over-confident; "99%" routinely means something
closer to 90%. Reporting an uncalibrated number as a probability to a farmer is
misleading.

**Temperature scaling** fits one scalar T on held-out validation data and
divides the logits by it. Because a positive scalar cannot change which logit is
largest, accuracy is provably unchanged — only the confidence values move. It is
the cheapest calibration method that works, needs one parameter, and cannot
degrade the classifier.

The banding into HIGH / MEDIUM / LOW is then meaningful rather than cosmetic:

| Band | Threshold | What the UI does |
|---|---|---|
| HIGH | ≥ 0.85 | Present the result normally |
| MEDIUM | ≥ 0.60 | Show the result, ask the user to compare alternatives |
| LOW | < 0.60 | Advise a clearer photo or an expert |

Both the raw and the calibrated reliability diagrams are rendered, so the
improvement is shown, not claimed.

---

## Why an out-of-distribution stage?

A softmax layer always returns a distribution over the 38 trained classes. Show
it a photograph of a keyboard and it will return one confidently. That is not a
bug in the model; it is what a closed-set classifier is.

Three scores are computed from the model's own output:

* **Maximum softmax probability** — simple, strong, but inflated by
  over-confidence.
* **Normalised entropy** — how spread the belief is across classes.
* **Free energy** `−logsumexp(logits)` — keeps the *absolute* logit magnitude,
  which softmax normalises away, so it separates near-OOD inputs better.

Thresholds are set to retain a target 95% true-positive rate on genuine leaves,
which makes the false-rejection cost explicit and tunable.

**The measured result is not flattering, and is reported anyway:** stand-alone
AUROC for these scores on the synthetic OOD proxies is 0.975 (max softmax
probability), 0.977 (normalised entropy) and 0.942 (free energy) on the
production model. On the from-scratch CNN baseline the same measurement gave
about 0.62-0.63 - barely better than chance - so the separation comes from the
pretrained backbone's calibration rather than from the scores themselves. Treat
both figures as upper bounds: synthetic proxies are easier to reject than a real
photograph of an unrelated subject.

---

## Why an image-quality gate *before* the model?

Because it catches a different failure than the OOD score, and it catches it
more cheaply and more informatively.

| Signal | Catches | Cost |
|---|---|---|
| Laplacian variance | Out-of-focus photographs | One convolution |
| Mean luminance | Under/over-exposure | One mean |
| Luminance std | Flat, contrast-free images | One std |
| **Lag-1 pixel autocorrelation** | Synthetic noise and non-photographs | Two correlations |
| HSV band coverage | Frames with no plant-like colour | One mask |

The autocorrelation check is the interesting one. Natural photographs obey
well-known image statistics: neighbouring pixels are strongly correlated,
typically above 0.9 even in detailed texture. Uniform noise has essentially
none. The Laplacian blur check *cannot* catch noise — noise scores extremely
high on sharpness — so before this check was added, uniform-noise images passed
the gate and were confidently classified.

Adding it roughly doubled the measured whole-pipeline OOD rejection rate, for a
small single-digit cost in genuine leaves asked for a retake. The exact figures
depend on which model is being served, so they are regenerated by
`training/calibrate_ood.py` and reported in
[`docs/results.md`](results.md#out-of-distribution-handling) rather than fixed
here. When first measured on the from-scratch CNN baseline, the numbers were 39%
-> 76% rejection at a 3.3% false-rejection cost. Repeating the measurement on the production model gives 95.0% rejection at a 2.0% cost; current figures are always in `docs/results.md`, which is generated.

Crucially the gate gives *actionable* feedback ("hold the camera steady, tap to
focus") where the OOD score can only say "I don't know". A user can fix a blurry
photo; they cannot fix a low softmax score.

---

## Why classical ML baselines?

Three reasons, all of them about intellectual honesty:

1. **They establish the floor.** If hand-crafted colour histograms reach 90%,
   then a CNN reaching 95% has bought five points, not ninety-five. Without the
   baseline there is no way to know how much of the performance came from deep
   learning and how much from the dataset being easy.
2. **They make "why CNN?" an empirical question.** The benchmark table answers
   it with numbers instead of received wisdom.
3. **They cost almost nothing.** Feature extraction is ~16 ms per image and the
   features are cached, so five model families train on the same matrix.

The features chosen are the ones a computer-vision practitioner would actually
have reached for: RGB and HSV histograms and colour moments (chlorosis is a
colour shift), Local Binary Patterns and GLCM Haralick statistics (lesion
micro-texture), and HOG (lesion borders and vein structure).

---

## Why compare many models?

Because "we used a CNN and got 96%" is not a result — it is an anecdote. A
comparison answers questions the single number cannot:

* Is the accuracy coming from the architecture or from the dataset?
* Does the extra capacity of ResNet50 buy anything over MobileNetV3 here?
* What does the proposed CBAM model cost in latency and size for what it gains?
* Which model should actually ship?

The last question is why `export_model.py` supports several selection criteria —
best macro F1, best accuracy, best accuracy-per-cost, fastest, smallest — with
the chosen criterion and its justification recorded in the export manifest and
shown in the UI. The production model is chosen on merit; CBAM remains the
*research* contribution whether or not it wins that selection.

---

## Why data augmentation?

To make the model invariant to things that do not change the diagnosis, and only
to those things. A leaf photographed upside-down has the same disease; a leaf
whose hue has been rotated 40° may not.

That constraint rules out several popular transforms. Vertical flips are kept
(there is no canonical "up" for a leaf on a bench); aggressive hue jitter is
rejected (hue *is* the signal); large occlusions are rejected (they can remove
the only lesion in the frame). Augmentation is applied to the training split
only — augmenting the validation set would measure the augmentation rather than
the model.

---

## Why train / validation / test?

Three splits exist because there are three distinct decisions to protect:

| Split | Used for | If you skip it |
|---|---|---|
| **Train** | Fitting weights | — |
| **Validation** | Early stopping, LR scheduling, checkpoint selection, temperature fitting | Those decisions are made on test data and the reported number is optimistic |
| **Test** | The number you report, computed once | You have no honest estimate at all |

Every choice that "looks at" the data is a form of fitting. Early stopping on
the test set fits the test set. In this project the official `valid/` directory
is the test set and is touched exactly once per model, after training is
complete; validation comes out of the training pool.

---

## How does inference actually work?

```
Upload
  → validate MIME type, extension and size; re-decode the bytes (rejects polyglots)
  → strip EXIF by re-encoding (removes GPS)
  → quality analysis on the original resolution
  → if an error-level issue: stop and explain; do not classify
  → Resize + CenterCrop + Normalize   (identical to validation-time preprocessing)
  → forward pass under no_grad on the model loaded at startup
  → logits ÷ T  →  calibrated softmax
  → OOD assessment on the raw logits
  → Grad-CAM / Grad-CAM++ / CBAM map (needs a backward pass; ~2× the latency)
  → knowledge-base lookup for the predicted class
  → persist the record; return the response
```

Two properties matter:

* **The model is loaded once, at process startup.** No request loads weights;
  nothing retrains. Training and serving share only the checkpoint file.
* **Preprocessing is the same code object used in validation.** Not a
  reimplementation — the same function, imported from the same module.

---

## Why a knowledge base rather than generated text?

Because a generated paragraph about a plant disease is exactly the kind of
plausible-sounding, unverifiable content that makes a tool dangerous. The
knowledge base is a curated JSON file with an explicit editorial policy:

* Only widely documented plant-pathology facts — causal organism, classic
  symptom picture, conditions that favour the disease, cultural management.
* **No product names, no application rates, no spray schedules.** Those are
  jurisdiction-specific and change with product registration.
* Every entry carries a disclaimer and points to local agricultural extension.
* The generator **fails loudly** if the dataset contains a class it has no entry
  for, so the knowledge base cannot silently fall out of sync with the model.
