# Viva preparation

Questions an examiner is likely to ask, with answers grounded in what this
project actually did. Where a number appears, it comes from
`artifacts/results/` — check `docs/results.md` for the current values before the
viva, since they update whenever the pipeline is re-run.

---

## Fundamentals

**Q. Explain CBAM in one minute.**

CBAM is two attention blocks applied in sequence to a feature map. The channel
branch asks *what* is important: it pools every feature map to one number using
both average and max pooling, pushes both through a shared bottleneck MLP, sums
them and applies a sigmoid, giving one gate per channel. The spatial branch then
asks *where*: it collapses the channel axis with average and max pooling into a
two-channel map, applies a 7×7 convolution and a sigmoid, giving one gate per
pixel. Both gates multiply the feature map. It adds under 1% to the parameter
count.

**Q. Why does CBAM use both average and max pooling when SE uses only average?**

They measure different things. Average pooling measures how *extensively* a
feature is present across the leaf; max pooling measures its *strongest*
evidence anywhere. A small but intense lesion has a low average and a high
maximum — average pooling alone would nearly miss it. Arm D of the ablation is
SE precisely so this difference is measured, not assumed.

**Q. Why channel first, then spatial?**

The original paper compared all three orderings and found sequential
channel→spatial best. Intuitively, refining which channels to trust first gives
the spatial branch a cleaner signal to localise within. This project did not
re-run that ordering comparison; it did run a placement study - which *stages*
of the network get a CBAM block - and that turned out to matter more than the
choice of mechanism. See the next question.

**Q. Which mattered more: what kind of attention, or where you put it?**

Where. Varying the mechanism across the five ablation arms moved macro F1 by at
most about half a point over the control, which only narrowly clears the measured
run-to-run floor. Varying the *placement* of one fixed CBAM block moved it by
roughly a point - comfortably beyond that floor - and in opposite directions:
attention on the last two stages beat the no-attention control, while attention
on the last stage alone was slightly *worse* than no attention at all.

The mechanism is that the early stages of this network carry generic edge and
colour responses where there is little to gate, and gating them discards signal
the later layers still need. By the last two stages the features are specific
enough that suppressing the irrelevant ones helps. Confining attention to the
final stage leaves too little depth for the reweighted features to be used.

The honest framing for the viva is that the ablation's headline question - does
CBAM beat a plain CNN - got a weak answer, while a question the study almost
treated as secondary got a clear one.

**Q. What is the difference between attention here and attention in a
Transformer?**

Transformer attention computes pairwise similarities between tokens (queries,
keys, values) and is quadratic in sequence length. CBAM computes a gate from
pooled statistics — it is a learned, input-dependent reweighting, not a pairwise
interaction. It is far cheaper and has no notion of "attending from position A
to position B".

**Q. Why 7×7 for the spatial convolution?**

Deciding whether a region is lesion or background needs surrounding context. A
3×3 kernel sees almost nothing at that stage of the network. The paper found 7×7
beat 3×3; this implementation restricts the choice to those two values.

---

## Methodology

**Q. Why did you not train every model on all 70,000 images?**

Because the available GPU is a 4 GB GTX 1650, and 18 architectures on 63,265
images was not feasible in the time available. The alternative — training each
model on whatever budget happened to be convenient — would have made the
comparison meaningless. Instead every model in the benchmark and the ablation
gets an *identical* fixed budget: the same 300-images-per-class subset, the same
image size, epoch count, optimiser and seed. Only the architecture changes. The
selected models are then retrained on the full split, and the `protocol` column
marks which regime produced each row so the two are never conflated.

**Q. Isn't the accuracy inflated because the dataset is pre-augmented?**

That is the right concern, and it is why the project checks. A perceptual hash
was computed for a stratified sample of every class in both the official train
and valid directories and collisions were counted: zero. That is a sampled
estimate, not proof, and pHash would not catch a rotated or flipped augmentation
of the same original — both limitations are stated in `docs/methodology.md`.

Separately, the *bigger* honesty issue is not leakage but domain: PlantVillage
images are single leaves on uniform backgrounds under controlled lighting.
Accuracy here says little about a field photograph.

**Q. Why is the official `valid/` directory your test set?**

Because a validation set that influences early stopping, learning-rate
scheduling, checkpoint selection and temperature fitting is no longer a clean
estimate of generalisation. Those decisions are all forms of fitting. So
validation is carved out of the official training pool, and the official `valid/`
directory is touched exactly once per model, after training completes.

**Q. Why did you not use class weighting or focal loss?**

Because the measured imbalance ratio is 1.23 with a coefficient of variation of
0.056 — the dataset is close to uniform. Re-weighting a near-uniform
distribution adds gradient noise without improving minority-class recall. All
four remedies (weights, focal loss, oversampling, balanced sampling) are
implemented and selectable; the decision rule in `prepare_splits.py` chose
"none" from the real counts and recorded its reasoning in `data/splits.json`.

**Q. Why PyTorch when the proposal said TensorFlow/Keras?**

Hardware constraint, and it is documented. TensorFlow's native-Windows GPU
support ended at version 2.10, which requires Python ≤ 3.10. The target machine
runs Python 3.13 on Windows, so a TensorFlow build here would have been
CPU-only, on a project whose GPU sweep already takes hours. PyTorch's CUDA build
works and was verified against the GTX 1650.
The architecture is framework-agnostic: all model code lives in
`backend/app/ml/` and is shared by training and serving, so a port would touch
one directory.

---

## Results and honesty

**Q. Your proposal predicted 98.6%. What did you actually get?**

Whatever `docs/results.md` says — it is generated from the experiment ledger, and
no number in this project is typed by hand. The benchmark numbers are lower than
a full-data run would give because they come from the fixed-budget subset by
design. If the measured figure is below the proposal's estimate, the estimate
was wrong, not the experiment.

**Q. Did CBAM actually help?**

Read the ablation verdict in `docs/ablation_study.md` - it is computed from the
five arms and states the delta explicitly. The threshold for calling a gain real
is not a rule of thumb: it is **measured**.

Several configurations are, by construction, trained twice. `abl_cnn_none` is the
same network and schedule as `cnn_baseline`; `abl_cnn_se` the same as `cnn_se`;
and `cnn_cbam`, `abl_cnn_cbam` and `place_cbam_all` are all the same network
again, because `abl_cnn_cbam` leaves `attention_stages` at its default of "all".
The spread within each such group is run-to-run variation, and the largest spread
becomes the threshold.

Be precise about what that measures, because the obvious description is wrong.
Some of those repeats also ran with a different DataLoader worker count, and each
worker gets its own derived seed, so the augmentation stream and batch
composition differ too - they are not pure determinism checks. More importantly,
every repeat shares an *initialisation*, whereas the arms being compared do not:
a CNN and a CNN+CBAM have different parameter shapes and draw different starting
weights from the same seed. So the measured floor is a **lower bound** on the
noise relevant to the comparison, not a significance test. The report says so,
and says what would fix it: retraining one architecture at several seeds.

The spreads were also far from consistent - 0.04, 0.22 and 0.35 points across the
three groups - which is itself the argument for using the largest rather than the
average. A single close-agreeing repeat would have understated the floor by
roughly eight times, and did, until the second group finished.

Four outcomes are possible and all four are reported honestly:

* a gain more than twice the measured floor - reported as supported;
* a gain that clears the floor only narrowly - reported as *suggestive rather
  than established*, with the multiple quoted;
* a positive delta smaller than the floor - reported as within run-to-run
  variation;
* no gain or a regression - reported as such, with the likely reason.

If an examiner asks "how do you know that difference is real?", this is the
answer: the experiment measures its own noise floor, states what that measurement
does and does not cover, and grades its language accordingly.

If CBAM does not clearly help here, the most plausible explanation is the
dataset: PlantVillage leaves sit on uniform backgrounds, which is exactly the
regime where spatial attention has the least work to do. There is very little
background for it to suppress. That reading is supported by the placement study,
where the same block helped or hurt depending only on the depth it was inserted
at.

**Q. Why is your out-of-distribution AUROC only ~0.63? Isn't that bad?**

It is, and it is a well-documented property of softmax-based OOD detection
rather than a bug in this implementation — deep networks are famously confident
on inputs unlike anything they were trained on. That finding is *why* the system
does not rely on the OOD score alone. The image-quality gate runs first, and the
end-to-end pipeline rejects roughly 76% of the OOD proxies while asking for a
retake on about 3% of genuine leaves. Both figures are measured and reported.

**Q. What would you do with more compute?**

In priority order: (1) train every benchmark arm on the full split so the
comparison is at production scale; (2) run each ablation arm with three to five
seeds so differences can be tested for significance instead of being called
inconclusive; (3) collect or source real field imagery, which is the largest gap
between this and a deployable tool; (4) calibrate OOD against a real non-leaf
corpus rather than synthetic proxies.

---

## Engineering

**Q. Where does the model live at runtime, and when is it loaded?**

Once, in the FastAPI lifespan startup hook, into a process-wide registry. Every
request reuses that in-memory model. The API cannot train, cannot read the
dataset and cannot compute a metric — it reads a checkpoint and some JSON
artefacts. Which checkpoint is decided by `models/exported/production.json`,
written by the export step from measured results.

**Q. How do you guarantee training and inference preprocess identically?**

They import the same function from the same module. `app/ml/transforms.py` is
used by the training scripts and by the serving code; there is no second
implementation to drift. A duplicated transform is one of the most common silent
accuracy killers in deployed vision systems.

**Q. What happens if someone uploads a photo of a cat?**

The quality gate passes it (it is a sharp, colourful photograph). The model
produces a distribution over the 38 leaf classes. The OOD stage then examines
the maximum softmax probability and the normalised entropy against thresholds
calibrated for a 95% true-positive rate on real leaves. If either fails, the
response is `status: out_of_distribution` with the message "Unable to
confidently identify this image", and the UI presents it as such rather than as
a diagnosis. Honestly: a cat photo is *not guaranteed* to be caught — see the
AUROC question above.

**Q. What happens to a blurry photo?**

The quality gate catches it before inference and returns
`status: poor_quality` with specific advice ("hold the camera steady, tap to
focus"). No classification is attempted, because a classification the system
already knows to be unreliable is worse than no classification.

**Q. How is the uploaded image handled securely?**

Content type, extension and size are validated; the bytes are then re-decoded
with Pillow to confirm the *actual* format rather than trusting the declared one
(this rejects polyglot files); pixel count is capped to prevent decompression
bombs; the stored filename is a server-generated UUID, never derived from user
input, which eliminates path traversal; and the image is re-encoded before being
saved, which strips EXIF — including GPS coordinates.

**Q. Why is mixed precision disabled?**

Because it was measured, not assumed. A micro-benchmark at startup found fp16
running at 0.24× the speed of fp32 on this GPU — the GTX 1650's TU117 chip has
no tensor cores and its fp16 convolutions take a slow path. Enabling AMP would
have roughly tripled total training time. The check enables AMP only when the
measured speed-up exceeds 1.15×, so on tensor-core hardware it turns itself back
on.

**Q. How do you know your ablation differences are not noise?**

By measuring the noise — but be precise about *which* noise, because the obvious
measurement is the wrong one.

Two configurations are trained twice by construction, so their disagreement is
run-to-run noise. That number is small, and it is tempting to use it as the
significance threshold. It would be wrong to: those two runs share an
initialisation, whereas the arms being compared do not. A CNN and a CNN+CBAM have
different parameter shapes, so they draw different starting weights from the same
seed, and every delta between them carries initialisation and data-order variance
that the duplicate-pair measurement excludes by construction. Using it as a
threshold would make ordinary variation look like a finding.

So the report gives two floors. The duplicate pairs bound cuDNN scheduling jitter
and are labelled a lower bound. The threshold that actually grades the verdict
comes from retraining one architecture at several seeds, where everything that
differs between two architectures varies except the architecture. Where that has
not been run, `docs/ablation_study.md` says so explicitly and marks its
conclusions as weaker rather than quietly borrowing the narrower number.

**Q. What was the hardest bug?**

Training runs that hung with the GPU idle. The cause was Windows-specific: each
spawned DataLoader worker re-imports the CUDA build of torch, which reserves
about 1 GB of *commit charge* per worker. With 7.35 GB of RAM the system commit
limit was reached, workers died during import with `WinError 1455` ("the paging
file is too small"), and the parent process blocked forever on an empty queue —
with no error message, because the failure was in a child. The fix is two-part:
budget the worker count against available RAM, and detect that specific failure
and retry with in-process loading rather than hanging.

---

## Questions worth asking back

If the examiner asks "what would you change", these are the honest answers:

* The benchmark subset is a compromise forced by hardware, not a design
  preference. It makes the comparison fair but the absolute numbers pessimistic,
  and rankings can shift with more data — small models are favoured in low-data
  regimes.
* Single-seed results cannot support significance claims. The report says
  "inconclusive" where a multi-seed study would say "significant" or "not
  significant", and that is a real limitation, not a hedge.
* The system is trained on laboratory imagery and evaluated on laboratory
  imagery. Every accuracy figure in this project should be read with that
  qualifier attached.
* The disease knowledge base is deliberately conservative — no product names, no
  rates, no schedules — because those are jurisdiction-specific and change with
  registration. That makes it less immediately actionable and more defensible.
