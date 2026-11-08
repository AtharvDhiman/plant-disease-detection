"""Generate a comprehensive PDF project report for the Plant Disease Detection System.

Builds an executive-level engineering and research report with:
- System overview, problem statement, and agricultural context
- Dataset anatomy, leakage analysis, and split protocol
- System architecture (offline training vs. online serving, request sequence)
- Model stack (Classical ML, Custom CNN, Attention mechanisms, Transfer Learning)
- Training methodology, hardware profiling, and optimization
- Benchmark results & ablation study with exact ledger metrics
- Calibration, OOD detection, image quality, and explainability
- Complete tool and technology stack
- Production deployment on Render (512 MB memory engineering)
- Real-world domain shift analysis and roadmap
- Embedded high-resolution plots, diagrams, and UI screenshots

Renders to PDF via Playwright Chromium.
"""
from __future__ import annotations

import base64
import json
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

def image_to_base64(path: Path) -> str:
    if not path.exists():
        return ""
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    ext = path.suffix.lower().replace(".", "")
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"
    return f"data:{mime};base64,{b64}"

def load_json(path: Path, default=None):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default

def generate_report():
    print("Loading project artifacts...")
    summary = load_json(PROJECT_ROOT / "artifacts/results/summary.json", {})
    ablation = load_json(PROJECT_ROOT / "artifacts/results/ablation_summary.json", {})
    classical = load_json(PROJECT_ROOT / "artifacts/results/classical_results.json", {})
    serving_json = load_json(PROJECT_ROOT / "models/exported/serving.json", {})
    dataset_report = load_json(PROJECT_ROOT / "artifacts/dataset_report.json", {})
    splits = load_json(PROJECT_ROOT / "data/splits.json", {})
    domain_shift = load_json(PROJECT_ROOT / "artifacts/results/domain_shift.json", {})
    field_comparison = load_json(PROJECT_ROOT / "artifacts/results/field_comparison.json", {})
    ood_points = load_json(PROJECT_ROOT / "artifacts/results/ood_operating_points.json", {})
    footprint = load_json(PROJECT_ROOT / "artifacts/results/serving_footprint.json", {})
    manifest = load_json(PROJECT_ROOT / "models/exported/production.json", {})

    # Everything below describes the model the application serves, read from its
    # manifest so the report cannot quote a model that is no longer deployed.
    deployed = serving_json.get("test_metrics", {})
    dep_label = serving_json.get("label", "the deployed model")
    dep_protocol = manifest.get("protocol", "production")
    dep_id = manifest.get("experiment_id", "")
    dep_acc = f"{deployed.get('accuracy', 0) * 100:.2f}%"
    dep_f1 = f"{deployed.get('f1_macro', 0):.4f}"
    dep_f1_pct = f"{deployed.get('f1_macro', 0) * 100:.2f}%"
    dep_top3 = f"{deployed.get('top3_accuracy', 0) * 100:.2f}%"
    dep_ece_raw = f"{deployed.get('ece', 0) * 100:.2f}%"
    dep_ece_cal = f"{deployed.get('ece_after_temperature', 0) * 100:.2f}%"
    dep_ece_gain = (f"{deployed['ece'] / deployed['ece_after_temperature']:.0f}"
                    if deployed.get("ece_after_temperature") else "—")
    dep_t = f"{serving_json.get('temperature', 1):.4f}"
    dep_size = f"{serving_json.get('size_mb', 0):.1f}"
    dep_params = f"{serving_json.get('parameters', 0) / 1e6:.2f}M"
    publication_date = datetime.now().strftime("%B %Y")
    dep_latency = f"{manifest.get('inference_ms', 0):.1f}"
    dep_msp = serving_json.get("ood_thresholds", {}).get("msp_min")
    dep_entropy = serving_json.get("ood_thresholds", {}).get("entropy_max")
    inference_ms = f"{footprint.get('inference_ms_median', 0):.0f}"
    pipeline_ms = f"{footprint.get('pipeline_ms_median_no_explain', 0):.0f}"
    resident_mb = f"{footprint.get('resident_mb_after_requests', 0):.0f}"

    def pct1(value):
        return "—" if value is None else f"{value * 100:.1f}%"

    # Behaviour on field-like photos, from the measured files.
    shift_labels = {
        "original": ("Original Clean", "Lab control condition"),
        "background": ("Field Background", "Procedural soil, mulch and clutter backdrop"),
        "angle": ("Camera Angle", "Rotation and perspective squash"),
        "lighting": ("Harsh Lighting", "Directional sun gradient and contrast"),
        "compression": ("Compression Artifacts", "Severe JPEG quantisation (Q 14-24)"),
        "distance": ("Distance / Scale", "Leaf at about a third of the frame"),
        "combined": ("Combined Field Simulation", "Background + angle + lighting + JPEG"),
    }
    shift_rows = ""
    for key, (name, how) in shift_labels.items():
        entry = (domain_shift.get("conditions") or {}).get(key)
        if not entry:
            continue
        shade = ' style="background-color: #fff1f2;"' if key == "combined" else ""
        shift_rows += (f"<tr{shade}><td><strong>{name}</strong></td><td>{how}</td>"
                       f"<td class='numeric'>{pct1(entry.get('accuracy'))}</td>"
                       f"<td class='numeric'>{pct1(entry.get('refused_rate'))}</td></tr>\n")
    shift_n = domain_shift.get("images_per_condition", "—")

    comparison_rows = ""
    comparison_order = {"production": 0, "field": 1, "field_long": 2}
    for alias, result in sorted(
            (field_comparison.get("results") or {}).items(),
            key=lambda kv: (comparison_order.get(kv[0].partition("@")[2], 9), kv[0])):
        cells = ""
        for key in shift_labels:
            c = result["conditions"][key]
            cells += f"<td class='numeric'>{c['correct'] / c['n'] * 100:.0f}% ({c['ok_wrong']})</td>"
        serving_row = result.get("experiment_id") == dep_id
        shade = ' style="background-color: #ecfdf5; font-weight: 600;"' if serving_row else ""
        comparison_rows += (f"<tr{shade}><td>{result['label']}{' (serving)' if serving_row else ''}</td>"
                            f"<td class='numeric'>{result['lab_test_accuracy'] * 100:.2f}%</td>{cells}</tr>\n")
    comparison_n = next(iter((field_comparison.get("results") or {}).values()), {}) \
        .get("conditions", {}).get("original", {}).get("n", "—")

    # Real photographs: the PlantDoc held-out test split, the "not sure"
    # threshold chosen on real validation photos, and the data sources.
    real_eval = load_json(PROJECT_ROOT / "artifacts/results/real_photo_eval.json", {})
    real_op = load_json(PROJECT_ROOT / "artifacts/results/real_operating_point.json", {})
    ensemble = (load_json(PROJECT_ROOT / "artifacts/results/ensemble_max_r50plus.json", {})
                or load_json(PROJECT_ROOT / "artifacts/results/ensemble_eval.json", {}))
    real_n = real_eval.get("n", "—")
    real_rows = ""
    order = ["production", "field_long", "field_long_resnet50", "field_real", "field_real_all",
             "field_real_plus", "field_real_plus_resnet50"]
    trained_on = {"production": "lab only", "field_long": "lab only (field augmentation)",
                  "field_long_resnet50": "lab only (field augmentation)",
                  "field_real": "lab + 1 real-photo set, 6 ep", "field_real_all": "lab + 3 sets, 6 ep",
                  "field_real_plus": "lab + 5 sets, 10 ep", "field_real_plus_resnet50": "lab + 5 sets, 10 ep"}
    for experiment_id, r in sorted((real_eval.get("results") or {}).items(),
                                   key=lambda kv: order.index(kv[1]["protocol"])
                                   if kv[1]["protocol"] in order else 99):
        live = experiment_id == dep_id
        shade = ' style="background-color: #ecfdf5; font-weight: 600;"' if live else ""
        low, high = r["top1_ci95"]
        real_rows += (f"<tr{shade}><td>{r['label']}{' (serving)' if live else ''}</td>"
                      f"<td>{trained_on.get(r['protocol'], r['protocol'])}</td>"
                      f"<td class='numeric'>{r['top1'] * 100:.1f}% ({low * 100:.0f}–{high * 100:.0f})</td>"
                      f"<td class='numeric'>{r['top3'] * 100:.1f}%</td>"
                      f"<td class='numeric'>{r['lab_test_accuracy'] * 100:.2f}%</td></tr>\n")
    live_real = (real_eval.get("results") or {}).get(dep_id, {})
    real_top1 = f"{live_real.get('top1', 0) * 100:.1f}%"
    real_top3 = f"{live_real.get('top3', 0) * 100:.1f}%"
    old_real = next((r for r in (real_eval.get("results") or {}).values()
                     if r["protocol"] == "field_long"), {})
    old_real_top1 = f"{old_real.get('top1', 0) * 100:.1f}%"

    op_rows = ""
    chosen = real_op.get("chosen") or {}
    for r in real_op.get("sweep", []):
        if round(r["msp_min"] * 100) % 10 and abs(r["msp_min"] - chosen.get("msp_min", -1)) > 1e-6:
            continue
        mark = abs(r["msp_min"] - chosen.get("msp_min", -1)) < 1e-6
        shade = ' style="background-color: #ecfdf5; font-weight: 600;"' if mark else ""
        op_rows += (f"<tr{shade}><td class='numeric'>{r['msp_min']:.2f}{' (in force)' if mark else ''}</td>"
                    f"<td class='numeric'>{r['coverage'] * 100:.1f}%</td>"
                    f"<td class='numeric'>{r['precision'] * 100:.1f}%</td>"
                    f"<td class='numeric'>{r['shown_wrong'] * 100:.1f}%</td></tr>\n")
    op_photos = real_op.get("photos", "—")
    op_wrong_now = f"{chosen.get('shown_wrong', 0) * 100:.1f}%"
    op_wrong_before = next((f"{r['shown_wrong'] * 100:.1f}%" for r in real_op.get("sweep", [])
                            if abs(r["msp_min"] - 0.15) < 1e-6), "—")

    source_rows = ""
    for name, info in (splits.get("external_sources") or {}).items():
        source_rows += (f"<tr><td>{name}</td><td>{info.get('licence')}</td>"
                        f"<td>{info.get('citation')}</td></tr>\n")

    ensemble_rows = ""
    for split, info in (ensemble.get("splits") or {}).items():
        cells = "".join(f"<td class='numeric'>{info['scores'][k]['top1'] * 100:.1f}% / "
                        f"{info['scores'][k]['top3'] * 100:.1f}%</td>"
                        for k in [*ensemble.get("experiments", []), "ensemble"])
        ensemble_rows += f"<tr><td>{split} ({info['n']})</td>{cells}</tr>\n"

    # Collect images as base64
    img_compare_acc = image_to_base64(PROJECT_ROOT / "artifacts/plots/compare_accuracy.png")
    img_ablation = image_to_base64(PROJECT_ROOT / "artifacts/plots/ablation.png")
    img_placement = image_to_base64(PROJECT_ROOT / "artifacts/plots/cbam_placement.png")
    img_tradeoff = image_to_base64(PROJECT_ROOT / "artifacts/plots/tradeoff_size_accuracy.png")
    img_reliability = image_to_base64(PROJECT_ROOT / "artifacts/plots" / f"{dep_id}_reliability.png")
    img_dist_train = image_to_base64(PROJECT_ROOT / "artifacts/plots/class_distribution_train.png")
    img_cm = image_to_base64(PROJECT_ROOT / "artifacts/confusion_matrices" / f"{dep_id}_test.png")
    
    img_home = image_to_base64(PROJECT_ROOT / "docs/screenshots/home.png")
    img_dashboard = image_to_base64(PROJECT_ROOT / "docs/screenshots/dashboard.png")

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Plant Disease Detection System — Comprehensive Engineering & Research Report</title>
<style>
  @page {{
    size: A4;
    margin: 20mm 16mm;
  }}
  * {{
    box-sizing: border-box;
  }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    color: #1e293b;
    background-color: #ffffff;
    line-height: 1.55;
    font-size: 10pt;
    margin: 0;
    padding: 0;
  }}

  /* Headings */
  h1, h2, h3, h4, h5 {{
    color: #0f172a;
    font-weight: 700;
    margin-top: 1.2em;
    margin-bottom: 0.5em;
    page-break-after: avoid;
  }}
  h1 {{ font-size: 22pt; line-height: 1.2; color: #14532d; }}
  h2 {{ font-size: 14pt; border-bottom: 2px solid #e2e8f0; padding-bottom: 4px; margin-top: 1.8em; color: #166534; }}
  h3 {{ font-size: 11pt; margin-top: 1.2em; color: #1e293b; }}
  h4 {{ font-size: 10pt; color: #475569; }}

  p {{
    margin: 0.5em 0;
    text-align: justify;
  }}

  /* Page breaks */
  .page-break {{
    page-break-after: always;
  }}
  .avoid-break {{
    page-break-inside: avoid;
  }}

  /* Cover Page */
  .cover-page {{
    min-height: 90vh;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    padding: 40px 20px 20px 20px;
  }}
  .cover-badge {{
    display: inline-block;
    background-color: #dcfce7;
    color: #15803d;
    font-size: 9pt;
    font-weight: 700;
    padding: 4px 12px;
    border-radius: 9999px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 16px;
    border: 1px solid #bbf7d0;
  }}
  .cover-title {{
    font-size: 28pt;
    font-weight: 800;
    color: #14532d;
    line-height: 1.15;
    margin: 0 0 12px 0;
  }}
  .cover-subtitle {{
    font-size: 13pt;
    color: #475569;
    font-weight: 400;
    line-height: 1.45;
    max-width: 650px;
    margin-bottom: 30px;
  }}
  .cover-metrics {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin: 30px 0;
  }}
  .metric-card {{
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 12px;
    text-align: center;
  }}
  .metric-card .val {{
    font-size: 18pt;
    font-weight: 800;
    color: #15803d;
    display: block;
  }}
  .metric-card .lbl {{
    font-size: 7.5pt;
    color: #64748b;
    text-transform: uppercase;
    font-weight: 600;
    letter-spacing: 0.05em;
  }}
  .metric-card .sub {{
    font-size: 7.5pt;
    color: #94a3b8;
    margin-top: 2px;
  }}

  .cover-preview {{
    width: 100%;
    max-height: 340px;
    object-fit: cover;
    border-radius: 8px;
    border: 1px solid #cbd5e1;
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    margin: 15px 0;
  }}

  .cover-meta {{
    border-top: 1px solid #e2e8f0;
    padding-top: 16px;
    display: flex;
    justify-content: space-between;
    font-size: 8.5pt;
    color: #64748b;
  }}
  .cover-meta strong {{
    color: #1e293b;
  }}

  /* Tables */
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0;
    font-size: 8pt;
  }}
  th, td {{
    padding: 6px 8px;
    border: 1px solid #e2e8f0;
    text-align: left;
  }}
  th {{
    background-color: #f1f5f9;
    color: #334155;
    font-weight: 700;
  }}
  tr:nth-child(even) {{
    background-color: #f8fafc;
  }}
  td.numeric, th.numeric {{
    text-align: right;
    font-variant-numeric: tabular-nums;
  }}
  .tag {{
    display: inline-block;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 7pt;
    font-weight: 600;
  }}
  .tag-green {{ background: #dcfce7; color: #166534; }}
  .tag-blue {{ background: #dbeafe; color: #1e40af; }}
  .tag-amber {{ background: #fef3c7; color: #92400e; }}
  .tag-purple {{ background: #f3e8ff; color: #6b21a8; }}
  .tag-gray {{ background: #f1f5f9; color: #475569; }}

  /* Cards & Callouts */
  .callout {{
    border-left: 4px solid #16a34a;
    background-color: #f0fdf4;
    padding: 10px 14px;
    margin: 12px 0;
    border-radius: 0 6px 6px 0;
    font-size: 8.5pt;
  }}
  .callout-title {{
    font-weight: 700;
    color: #15803d;
    margin-bottom: 3px;
  }}
  .callout-amber {{
    border-left-color: #f59e0b;
    background-color: #fffbeb;
  }}
  .callout-amber .callout-title {{
    color: #b45309;
  }}
  .callout-blue {{
    border-left-color: #3b82f6;
    background-color: #eff6ff;
  }}
  .callout-blue .callout-title {{
    color: #1d4ed8;
  }}

  /* Figures */
  .figure-box {{
    margin: 14px 0;
    text-align: center;
  }}
  .figure-img {{
    max-width: 100%;
    height: auto;
    border-radius: 6px;
    border: 1px solid #e2e8f0;
  }}
  .figure-caption {{
    font-size: 7.5pt;
    color: #64748b;
    margin-top: 4px;
    font-style: italic;
  }}

  .grid-2 {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin: 10px 0;
  }}

  /* Code block */
  pre {{
    background: #0f172a;
    color: #f8fafc;
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 7.5pt;
    overflow-x: auto;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    line-height: 1.4;
  }}
  code {{
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 8pt;
    background: #f1f5f9;
    padding: 1px 4px;
    border-radius: 3px;
    color: #0f172a;
  }}
  pre code {{
    background: transparent;
    padding: 0;
    color: inherit;
  }}

  /* Diagram Flow */
  .flow-step {{
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 8px 10px;
    margin: 6px 0;
    font-size: 8pt;
  }}
  .flow-step strong {{
    color: #166534;
  }}
</style>
</head>
<body>

<!-- ========================================================================= -->
<!-- COVER PAGE                                                                -->
<!-- ========================================================================= -->
<div class="cover-page">
  <div>
    <span class="cover-badge">Engineering & Research Whitepaper</span>
    <h1 class="cover-title">Plant Disease Detection System</h1>
    <div class="cover-subtitle">
      A Deep Learning Vision Architecture with Convolutional Block Attention Modules (CBAM),
      Statistical Calibration, Image Quality Gating, and Lightweight Production Deployment.
    </div>

    <div class="cover-metrics">
      <div class="metric-card">
        <span class="val">{dep_acc}</span>
        <span class="lbl">Test Accuracy</span>
        <span class="sub">17,572 held-out images</span>
      </div>
      <div class="metric-card">
        <span class="val">{dep_f1}</span>
        <span class="lbl">Macro F1 Score</span>
        <span class="sub">38 balanced classes</span>
      </div>
      <div class="metric-card">
        <span class="val">{dep_ece_cal}</span>
        <span class="lbl">Calibrated ECE</span>
        <span class="sub">T={dep_t} scaling</span>
      </div>
      <div class="metric-card">
        <span class="val">~{resident_mb} MB</span>
        <span class="lbl">Serving Footprint</span>
        <span class="sub">ONNX Runtime (512MB RAM)</span>
      </div>
    </div>

    {"<img class='cover-preview' src='" + img_home + "' alt='Application Interface Preview'>" if img_home else ""}
  </div>

  <div class="cover-meta">
    <div>
      <strong>Author / Maintainer:</strong> AtharvDhiman<br>
      <strong>Repository:</strong> AtharvDhiman/plant-disease-detection
    </div>
    <div style="text-align: right;">
      <strong>Target Platform:</strong> Render Cloud (Docker Multi-Stage)<br>
      <strong>Date of Publication:</strong> {publication_date}
    </div>
  </div>
</div>

<div class="page-break"></div>

<!-- ========================================================================= -->
<!-- 1. EXECUTIVE SUMMARY & PROBLEM STATEMENT                                 -->
<!-- ========================================================================= -->
<h2>1. Executive Summary & Problem Context</h2>

<p>
Crop loss caused by botanical pathogens and pest infestations accounts for over <strong>20% to 40% of global agricultural yield reduction annually</strong>, translating to hundreds of billions of dollars in economic damages and endangering food security worldwide. Traditional agricultural diagnosis relies on manual visual inspection by agronomy specialists or molecular laboratory testing (e.g., Polymerase Chain Reaction, ELISA). However, in rural agricultural zones and smallholder farming environments, access to plant pathologists is scarce, expensive, and slow.
</p>

<p>
This project develops an <strong>end-to-end, trustworthy, production-grade deep learning system</strong> capable of accurately diagnosing <strong>38 distinct conditions across 14 vital crop species</strong> from standard photographs of leaf surfaces. Unlike academic prototypes that function merely as black-box classification scripts, this system incorporates a multi-tier engineering pipeline:
</p>

<ul>
  <li><strong>Image Quality Validation Gate:</strong> Automatically inspects uploaded images for motion blur, severe underexposure, overexposure, and lack of biological plant material before executing neural inference.</li>
  <li><strong>State-of-the-Art Deep Vision:</strong> Evaluates 18 architectures comprising handcrafted classical machine learning, custom convolutional neural networks with <strong>Convolutional Block Attention Modules (CBAM)</strong>, and modern transfer learning backbones (EfficientNet, ResNet, DenseNet, MobileNet).</li>
  <li><strong>Trust & Statistical Calibration:</strong> Incorporates post-hoc Temperature Scaling to align predictive probabilities with empirical error rates, dropping Expected Calibration Error (ECE) from <strong>{dep_ece_raw} to {dep_ece_cal}</strong>.</li>
  <li><strong>Out-of-Distribution (OOD) Rejection:</strong> Calculates joint metrics (Max Softmax Probability, Predictive Entropy, and Free Energy) calibrated to a 95% True Positive Rate, ensuring non-leaf or unrecognizable subjects are safely refused rather than hallucinating high-confidence misdiagnoses.</li>
  <li><strong>Explainable AI (XAI):</strong> Generates visual saliency maps via Grad-CAM, Grad-CAM++, and forward-pass Occlusion Sensitivity, pinpointing the specific pathological lesions that drove the prediction.</li>
  <li><strong>Resource-Constrained Cloud Deployment:</strong> Engineered specifically to operate within <strong>Render's 512 MB Free Tier</strong> by exporting PyTorch checkpoints to ONNX Runtime graphs, reducing memory footprint by ~78% (from 657 MB to ~{resident_mb} MB).</li>
</ul>

<div class="callout callout-blue avoid-break">
  <div class="callout-title">Key Architectural Achievement</div>
  The deployed model ({dep_label}, fine-tuned on eight real-photo datasets) achieves <strong>{dep_acc} accuracy, {dep_f1_pct} Macro F1 and {dep_top3} Top-3 accuracy</strong> on 17,572 held-out laboratory images and &mdash; the number that matters for a phone or web photo &mdash; <strong>{real_top1} on {real_n} held-out real photographs</strong> (top-3 {real_top3}), up from {old_real_top1} for the laboratory-trained model. It runs in ~{inference_ms} ms per image (~{pipeline_ms} ms for the whole pipeline) on a single CPU thread without a GPU (Section 10).
</div>

<!-- ========================================================================= -->
<!-- 2. DATASET ANATOMY & LEAKAGE SAFEGUARDS                                  -->
<!-- ========================================================================= -->
<h2>2. Dataset Anatomy & Curation Protocol</h2>

<p>
The system is built upon the <code>vipoooool/new-plant-diseases-dataset</code> corpus, an augmented derivative of the landmark PlantVillage dataset. The corpus contains <strong>87,867 total images</strong> across 38 classes and 14 agricultural plant species (Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato).
</p>

<div class="avoid-break">
  <h3>Dataset Geometry & Imbalance Analysis</h3>
  <table>
    <thead>
      <tr>
        <th>Property</th>
        <th>Measured Specification</th>
        <th>Engineering Implication</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>Total Volume</strong></td>
        <td>87,867 images (70,295 official train, 17,572 official valid)</td>
        <td>Ample visual data across all disease manifestations.</td>
      </tr>
      <tr>
        <td><strong>Class Cardinality</strong></td>
        <td>38 classes (12 healthy, 26 diseased conditions)</td>
        <td>Requires macro-averaged metrics to prevent majority bias.</td>
      </tr>
      <tr>
        <td><strong>Spatial Resolution</strong></td>
        <td>256 &times; 256 pixels, RGB 3-channel uniform</td>
        <td>Standardized dimension avoids resizing distortion.</td>
      </tr>
      <tr>
        <td><strong>Imbalance Ratio</strong></td>
        <td>1.23 (Min class: 1,520; Max class: 1,870)</td>
        <td>Near-uniform distribution (Coefficient of Variation: 0.056).</td>
      </tr>
      <tr>
        <td><strong>Imbalance Strategy</strong></td>
        <td>Standard Cross-Entropy (No reweighting needed)</td>
        <td>Reweighting balanced data injects gradient variance without gain.</td>
      </tr>
      <tr>
        <td><strong>File Integrity</strong></td>
        <td>0 corrupt files detected across exhaustive audit</td>
        <td>Safe for deterministic tensor conversion.</td>
      </tr>
    </tbody>
  </table>
</div>

<div class="avoid-break">
  <h3>Three-Way Partitioning Protocol & Data Leakage Prevention</h3>
  <p>
  A major vulnerability in visual deep learning benchmarks is data leakage—where identical or augmented copies of images appear in both training and evaluation sets. Furthermore, relying on an official validation set for both early stopping and final reporting produces optimistic bias. We enforced an absolute <strong>three-way split protocol</strong>:
  </p>
  <ul>
    <li><strong>Training Set (63,265 images, 90% of official train):</strong> Used strictly for backpropagation and parameter updating.</li>
    <li><strong>Validation Set (7,030 images, 10% of official train):</strong> Carved out via stratified sampling; dedicated exclusively to hyperparameter tuning, checkpoint selection, early stopping (patience=3), and post-hoc temperature fitting.</li>
    <li><strong>Held-Out Test Set (17,572 images, official valid):</strong> Held completely blind until the model was exported. Never touched during model optimization.</li>
  </ul>
  <p>
  <strong>Perceptual Hash (pHash) Verification:</strong> A stratified perceptual hash (8&times;8 DCT) check across 2,280 samples confirmed <strong>0 duplicate collisions</strong> between the training and validation boundaries, confirming data integrity.
  </p>
</div>

{"<div class='figure-box avoid-break'><img class='figure-img' style='max-height: 230px;' src='" + img_dist_train + "' alt='Class Distribution'><div class='figure-caption'>Figure 1: Stratified Class Distribution across all 38 Target Classes in the Training Split.</div></div>" if img_dist_train else ""}

<div class="page-break"></div>

<!-- ========================================================================= -->
<!-- 3. SYSTEM ARCHITECTURE & END-TO-END WORKFLOW                             -->
<!-- ========================================================================= -->
<h2>3. System Architecture & Request Workflow</h2>

<p>
The application enforces strict decoupling between <strong>Offline Training/Research</strong> and <strong>Online Serving</strong>. They share core ML definitions (transforms, quality checks) but communicate purely via immutable disk artifacts (checkpoints, ONNX graphs, JSON manifests). The online FastAPI server contains zero training routines, ensuring low memory and high reliability.
</p>

<div class="avoid-break">
  <h3>End-to-End Prediction Lifecycle</h3>
  <p>Every inference request submitted to <code>POST /api/predict</code> executes through a deterministic 7-stage pipeline:</p>

  <div class="flow-step">
    <strong>Stage 1: Transport & Payload Sanitization</strong><br>
    Validates file MIME type (JPEG, PNG, WebP), enforces a 10 MiB payload limit, strips EXIF metadata to preserve user privacy, and decodes the image in memory via Pillow.
  </div>

  <div class="flow-step">
    <strong>Stage 2: Image Quality Analysis Pre-Check</strong><br>
    Computes OpenCV variance of Laplacian (sharpness), mean luminance (exposure), contrast, green/brown color masking (vegetation presence), and lag-1 autocorrelation (noise coherence). If severe blur (&lt;45) or extreme darkness (&lt;35) is identified, inference halts and returns an actionable <code>poor_quality</code> diagnosis.
  </div>

  <div class="flow-step">
    <strong>Stage 3: Deterministic Preprocessing</strong><br>
    Resizes image aspect ratio preserving the shorter side to 255 px, applies a 224&times;224 center crop, converts to float32 tensor in range [0, 1], and standardizes using ImageNet mean (0.485, 0.456, 0.406) and std (0.229, 0.224, 0.225).
  </div>

  <div class="flow-step">
    <strong>Stage 4: ONNX Runtime Engine Forward Pass</strong><br>
    Feeds the (1, 3, 224, 224) tensor through the optimized ONNX session with 1 intra-op thread, computing raw uncalibrated class logits in ~{inference_ms} ms.
  </div>

  <div class="flow-step">
    <strong>Stage 5: Temperature-Scaled Calibration & OOD Scoring</strong><br>
    Scales raw logits by calibrated parameter <i>T = {dep_t}</i> and computes softmax probabilities. Simultaneously evaluates the raw logits against the "not sure" threshold (Max Softmax Probability &ge; {dep_msp}), chosen on real validation photographs so that answers shown as a diagnosis are right at least 90% of the time; below it the app shows the closest matches instead of a verdict (Section 10).
  </div>

  <div class="flow-step">
    <strong>Stage 6: Explainability & Heatmap Generation</strong><br>
    Computes visual saliency via exact CAM activation heads or forward-pass Occlusion Sensitivity (sliding 8&times;8 masking grid), generating a normalized color overlay.
  </div>

  <div class="flow-step">
    <strong>Stage 7: Agronomic Knowledge Enrichment & SQLite Storage</strong><br>
    Enriches the diagnosis with pathogen taxonomy, visual symptoms, biological treatment, chemical controls, and preventative measures from <code>data/disease_info.json</code>. Commits prediction metadata to local SQLite database with Write-Ahead Logging (WAL).
  </div>
</div>

<div class="avoid-break">
  <h3>Online vs. Offline Decoupling Architecture</h3>
  <table>
    <thead>
      <tr>
        <th>Subsystem</th>
        <th>Core Modules</th>
        <th>Responsibilities</th>
        <th>Strict Boundaries</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>Offline Training</strong></td>
        <td><code>training/</code>, <code>scripts/</code></td>
        <td>Dataset audit, manifests, PyTorch training loops, hyperparameter tuning, calibration, ONNX export.</td>
        <td>Never imported by the online web service.</td>
      </tr>
      <tr>
        <td><strong>Disk Interface</strong></td>
        <td><code>models/exported/</code>, <code>artifacts/</code></td>
        <td>ONNX graphs, JSON manifests, experiment ledgers, evaluation metrics, precomputed plots.</td>
        <td>Immutable contracts linking training to serving.</td>
      </tr>
      <tr>
        <td><strong>Online Backend</strong></td>
        <td><code>backend/app/</code></td>
        <td>FastAPI REST service, ONNX Runtime inference, image quality checks, SQLite persistence.</td>
        <td>Runs without PyTorch; zero training logic.</td>
      </tr>
      <tr>
        <td><strong>Client Frontend</strong></td>
        <td><code>frontend/src/</code></td>
        <td>React 19 Single Page App, Recharts data visualization, camera upload, interactive disease library.</td>
        <td>Zero metric calculation; consumes REST API.</td>
      </tr>
    </tbody>
  </table>
</div>

{"<div class='figure-box avoid-break'><img class='figure-img' style='max-height: 230px;' src='" + img_dashboard + "' alt='Analytics Dashboard'><div class='figure-caption'>Figure 2: Production Web Analytics & Diagnosis Dashboard (React 19 + Tailwind CSS).</div></div>" if img_dashboard else ""}

<div class="page-break"></div>

<!-- ========================================================================= -->
<!-- 4. THE MODEL STACK & ARCHITECTURAL DESIGNS                               -->
<!-- ========================================================================= -->
<h2>4. Model Architecture & Deep Learning Suite</h2>

<p>
To determine the optimal balance between diagnostic accuracy, computational complexity, inference latency, and memory footprint, we implemented and evaluated four distinct families of models under an identical experimental protocol:
</p>

<div class="avoid-break">
  <h3>Family 1: Hand-Crafted Feature Classical Machine Learning Baselines</h3>
  <p>
  Before adopting deep neural networks, we established rigorous classical baselines using <strong>115 engineered computer vision features</strong>:
  </p>
  <ul>
    <li><strong>Color Statistics (72 features):</strong> 24-bin normalized histograms across Hue, Saturation, and Value (HSV) color channels to detect chlorosis (yellowing) and necrosis (browning).</li>
    <li><strong>Texture Features (26 features):</strong> Gray-Level Co-occurrence Matrix (GLCM) Haralick texture metrics (contrast, dissimilarity, homogeneity, energy, correlation, ASM) at 4 spatial angles and 2 pixel distances.</li>
    <li><strong>Morphological Moments (7 features):</strong> Hu's 7 invariant image moments capturing contour and shape geometry.</li>
    <li><strong>Luminance Descriptors (10 features):</strong> Grayscale percentile distributions and spatial gradients.</li>
  </ul>
  <p>
  Classifiers trained on these features: <strong>Logistic Regression</strong> (L2 regularized), <strong>Linear Support Vector Classifier</strong>, <strong>Random Forest</strong> (200 trees, max depth 20), and <strong>XGBoost</strong> (150 estimators, learning rate 0.1).
  </p>
</div>

<div class="avoid-break">
  <h3>Family 2: Custom 4-Stage Convolutional Neural Network (Baseline)</h3>
  <p>
  A specialized deep convolutional network designed from scratch for leaf pathology:
  </p>
  <ul>
    <li><strong>Stage Structure (4 Stages):</strong> Each stage consists of <code>Conv2D(3x3) &rarr; BatchNorm &rarr; ReLU &rarr; Conv2D(3x3) &rarr; BatchNorm &rarr; ReLU &rarr; MaxPool2D(2x2)</code>. Filter channels double progressively: 32 &rarr; 64 &rarr; 128 &rarr; 256.</li>
    <li><strong>Classification Head:</strong> Global Average Pooling (GAP) &rarr; Dropout(0.4) &rarr; Linear(256) &rarr; ReLU &rarr; Dropout(0.2) &rarr; Linear(38 classes).</li>
    <li><strong>Total Parameters:</strong> 1,248,774 parameters (4.77 MB). Highly lightweight and fast (~2.1 ms CPU inference).</li>
  </ul>
</div>

<div class="avoid-break">
  <h3>Family 3: Convolutional Block Attention Module (CBAM)</h3>
  <p>
  CBAM sequentially infers 1D channel attention maps <i><b>M</b><sub>c</sub> &in; &reals;<sup>C&times;1&times;1</sup></i> and 2D spatial attention maps <i><b>M</b><sub>s</sub> &in; &reals;<sup>1&times;H&times;W</sup></i>:
  </p>
  <div style="background: #f8fafc; border: 1px solid #e2e8f0; padding: 8px 12px; border-radius: 6px; font-size: 8pt; margin: 8px 0;">
    <strong>Channel Attention Formulation:</strong><br>
    <b>M</b><sub>c</sub>(<b>F</b>) = &sigma;(MLP(AvgPool(<b>F</b>)) + MLP(MaxPool(<b>F</b>))) = &sigma;(<b>W</b><sub>1</sub>(<b>W</b><sub>0</sub>(<b>F</b><sup>c</sup><sub>avg</sub>)) + <b>W</b><sub>1</sub>(<b>W</b><sub>0</sub>(<b>F</b><sup>c</sup><sub>max</sub>)))<br>
    <em>Purpose:</em> Determines <strong>what</strong> features are important across channels with reduction ratio <i>r = 16</i>.<br><br>
    <strong>Spatial Attention Formulation:</strong><br>
    <b>M</b><sub>s</sub>(<b>F</b>') = &sigma;(<i>f</i><sup>7&times;7</sup>([AvgPool(<b>F</b>'); MaxPool(<b>F</b>')]))<br>
    <em>Purpose:</em> Determines <strong>where</strong> pathological lesion spots exist using a 7&times;7 convolutional receptive field.
  </div>
  <p>
  <strong>Placement Variants Explored:</strong> We tested CBAM on the last stage only, the last two stages, and across all 4 stages. We also compared CBAM against standalone Squeeze-and-Excitation (SE), channel attention alone, and spatial attention alone.
  </p>
</div>

<div class="avoid-break">
  <h3>Family 4: Pretrained Transfer Learning Backbones</h3>
  <p>
  Leveraged backbones pre-trained on ImageNet-1K with custom 2-stage fine-tuning:
  </p>
  <ul>
    <li><strong>EfficientNet-B0 (&plusmn; CBAM):</strong> Compound-scaled mobile architecture balancing depth, width, and resolution (4.06M params, 15.6 MB).</li>
    <li><strong>MobileNetV3-Large (&plusmn; CBAM):</strong> Hardware-aware NAS architecture with hard-swish activations and SE blocks (3.01M params, 11.6 MB).</li>
    <li><strong>ResNet-50 (&plusmn; CBAM):</strong> 50-layer residual network with bottleneck blocks (23.59M params, 90.2 MB).</li>
    <li><strong>DenseNet-121:</strong> Densely connected convolutional blocks facilitating feature reuse (6.99M params, 27.0 MB).</li>
  </ul>
</div>

<div class="page-break"></div>

<!-- ========================================================================= -->
<!-- 5. EXPERIMENTAL RESULTS & COMPREHENSIVE BENCHMARK                         -->
<!-- ========================================================================= -->
<h2>5. Experimental Results & Benchmark Evaluation</h2>

<p>
All 18 models were evaluated under an identical, controlled benchmark subset protocol (160&times;160 resolution, 12 epochs, AdamW, cosine decay, seed=42). Selected champions were subsequently trained on the full production dataset (224&times;224 resolution).
</p>

<div class="avoid-break">
  <h3>Comprehensive Architectural Benchmark Table</h3>
  <table>
    <thead>
      <tr>
        <th>Model Architecture</th>
        <th>Group</th>
        <th class="numeric">Accuracy</th>
        <th class="numeric">Macro F1</th>
        <th class="numeric">Top-3 Acc</th>
        <th class="numeric">Params</th>
        <th class="numeric">Size (MB)</th>
        <th class="numeric">Latency</th>
        <th class="numeric">ECE</th>
      </tr>
    </thead>
    <tbody>
      <tr style="background-color: #ecfdf5; font-weight: 600;">
        <td><strong>{dep_label} (field-trained, serving)</strong></td>
        <td><span class="tag tag-green">{dep_protocol}</span></td>
        <td class="numeric"><strong>{dep_acc}</strong></td>
        <td class="numeric"><strong>{dep_f1}</strong></td>
        <td class="numeric">{dep_top3}</td>
        <td class="numeric">{dep_params}</td>
        <td class="numeric">{dep_size} MB</td>
        <td class="numeric">{dep_latency} ms</td>
        <td class="numeric">{deployed.get('ece', 0):.4f}*</td>
      </tr>
      <tr style="background-color: #f0fdf4;">
        <td>EfficientNet-B0 (Production)</td>
        <td><span class="tag tag-green">Production</span></td>
        <td class="numeric"><strong>99.06%</strong></td>
        <td class="numeric"><strong>0.9904</strong></td>
        <td class="numeric">99.94%</td>
        <td class="numeric">4.06M</td>
        <td class="numeric">15.6 MB</td>
        <td class="numeric">19.5 ms</td>
        <td class="numeric">0.0500*</td>
      </tr>
      <tr style="background-color: #f0fdf4;">
        <td>EfficientNet-B0 + CBAM (Prod)</td>
        <td><span class="tag tag-green">Production</span></td>
        <td class="numeric">99.00%</td>
        <td class="numeric">0.9898</td>
        <td class="numeric">99.93%</td>
        <td class="numeric">4.26M</td>
        <td class="numeric">16.4 MB</td>
        <td class="numeric">17.8 ms</td>
        <td class="numeric">0.0520</td>
      </tr>
      <tr style="background-color: #f0fdf4;">
        <td>CBAM Last 2 Stages (Prod)</td>
        <td><span class="tag tag-green">Production</span></td>
        <td class="numeric">98.81%</td>
        <td class="numeric">0.9880</td>
        <td class="numeric">99.88%</td>
        <td class="numeric">1.26M</td>
        <td class="numeric">4.8 MB</td>
        <td class="numeric">5.7 ms</td>
        <td class="numeric">0.0428</td>
      </tr>
      <tr style="background-color: #f0fdf4;">
        <td>Custom CNN + CBAM (Prod)</td>
        <td><span class="tag tag-green">Production</span></td>
        <td class="numeric">98.42%</td>
        <td class="numeric">0.9842</td>
        <td class="numeric">99.84%</td>
        <td class="numeric">1.26M</td>
        <td class="numeric">4.8 MB</td>
        <td class="numeric">7.4 ms</td>
        <td class="numeric">0.0418</td>
      </tr>
      <tr>
        <td>ResNet-50 + CBAM</td>
        <td><span class="tag tag-blue">Benchmark</span></td>
        <td class="numeric">99.14%</td>
        <td class="numeric">0.9914</td>
        <td class="numeric">99.96%</td>
        <td class="numeric">24.11M</td>
        <td class="numeric">92.2 MB</td>
        <td class="numeric">15.9 ms</td>
        <td class="numeric">0.0612</td>
      </tr>
      <tr>
        <td>ResNet-50 (Baseline)</td>
        <td><span class="tag tag-blue">Benchmark</span></td>
        <td class="numeric">99.12%</td>
        <td class="numeric">0.9912</td>
        <td class="numeric">99.93%</td>
        <td class="numeric">23.59M</td>
        <td class="numeric">90.2 MB</td>
        <td class="numeric">14.2 ms</td>
        <td class="numeric">0.0614</td>
      </tr>
      <tr>
        <td>DenseNet-121</td>
        <td><span class="tag tag-blue">Benchmark</span></td>
        <td class="numeric">97.81%</td>
        <td class="numeric">0.9780</td>
        <td class="numeric">99.78%</td>
        <td class="numeric">6.99M</td>
        <td class="numeric">27.0 MB</td>
        <td class="numeric">33.5 ms</td>
        <td class="numeric">0.0617</td>
      </tr>
      <tr>
        <td>EfficientNet-B0 (Benchmark)</td>
        <td><span class="tag tag-blue">Benchmark</span></td>
        <td class="numeric">97.02%</td>
        <td class="numeric">0.9700</td>
        <td class="numeric">99.65%</td>
        <td class="numeric">4.06M</td>
        <td class="numeric">15.6 MB</td>
        <td class="numeric">19.5 ms</td>
        <td class="numeric">0.0721</td>
      </tr>
      <tr>
        <td>EfficientNet-B0 + CBAM (Bench)</td>
        <td><span class="tag tag-blue">Benchmark</span></td>
        <td class="numeric">96.82%</td>
        <td class="numeric">0.9681</td>
        <td class="numeric">99.63%</td>
        <td class="numeric">4.26M</td>
        <td class="numeric">16.4 MB</td>
        <td class="numeric">17.8 ms</td>
        <td class="numeric">0.0727</td>
      </tr>
      <tr>
        <td>Custom CNN + CBAM (Bench)</td>
        <td><span class="tag tag-blue">Benchmark</span></td>
        <td class="numeric">96.07%</td>
        <td class="numeric">0.9605</td>
        <td class="numeric">99.58%</td>
        <td class="numeric">1.26M</td>
        <td class="numeric">4.8 MB</td>
        <td class="numeric">7.4 ms</td>
        <td class="numeric">0.0418</td>
      </tr>
      <tr>
        <td>Custom CNN + SE</td>
        <td><span class="tag tag-blue">Benchmark</span></td>
        <td class="numeric">96.03%</td>
        <td class="numeric">0.9598</td>
        <td class="numeric">99.54%</td>
        <td class="numeric">1.26M</td>
        <td class="numeric">4.8 MB</td>
        <td class="numeric">4.5 ms</td>
        <td class="numeric">0.0626</td>
      </tr>
      <tr>
        <td>Custom CNN Baseline</td>
        <td><span class="tag tag-blue">Benchmark</span></td>
        <td class="numeric">95.83%</td>
        <td class="numeric">0.9578</td>
        <td class="numeric">99.50%</td>
        <td class="numeric">1.25M</td>
        <td class="numeric">4.8 MB</td>
        <td class="numeric">2.1 ms</td>
        <td class="numeric">0.0670</td>
      </tr>
      <tr>
        <td>MobileNetV3-Large</td>
        <td><span class="tag tag-blue">Benchmark</span></td>
        <td class="numeric">95.02%</td>
        <td class="numeric">0.9497</td>
        <td class="numeric">99.25%</td>
        <td class="numeric">3.01M</td>
        <td class="numeric">11.6 MB</td>
        <td class="numeric">13.6 ms</td>
        <td class="numeric">0.0957</td>
      </tr>
      <tr>
        <td>MobileNetV3-Large + CBAM</td>
        <td><span class="tag tag-blue">Benchmark</span></td>
        <td class="numeric">94.78%</td>
        <td class="numeric">0.9473</td>
        <td class="numeric">99.14%</td>
        <td class="numeric">3.12M</td>
        <td class="numeric">12.0 MB</td>
        <td class="numeric">14.2 ms</td>
        <td class="numeric">0.0901</td>
      </tr>
      <tr>
        <td>XGBoost (Handcrafted)</td>
        <td><span class="tag tag-amber">Classical</span></td>
        <td class="numeric">90.24%</td>
        <td class="numeric">0.9020</td>
        <td class="numeric">98.11%</td>
        <td class="numeric">—</td>
        <td class="numeric">2.4 MB</td>
        <td class="numeric">0.12 ms</td>
        <td class="numeric">—</td>
      </tr>
      <tr>
        <td>Random Forest (Handcrafted)</td>
        <td><span class="tag tag-amber">Classical</span></td>
        <td class="numeric">88.73%</td>
        <td class="numeric">0.8864</td>
        <td class="numeric">97.85%</td>
        <td class="numeric">—</td>
        <td class="numeric">14.1 MB</td>
        <td class="numeric">0.18 ms</td>
        <td class="numeric">—</td>
      </tr>
      <tr>
        <td>Linear SVM (Handcrafted)</td>
        <td><span class="tag tag-amber">Classical</span></td>
        <td class="numeric">87.96%</td>
        <td class="numeric">0.8778</td>
        <td class="numeric">96.01%</td>
        <td class="numeric">—</td>
        <td class="numeric">0.4 MB</td>
        <td class="numeric">0.09 ms</td>
        <td class="numeric">—</td>
      </tr>
      <tr>
        <td>Logistic Regression (Handcrafted)</td>
        <td><span class="tag tag-amber">Classical</span></td>
        <td class="numeric">87.24%</td>
        <td class="numeric">0.8719</td>
        <td class="numeric">97.15%</td>
        <td class="numeric">—</td>
        <td class="numeric">0.1 MB</td>
        <td class="numeric">0.02 ms</td>
        <td class="numeric">—</td>
      </tr>
    </tbody>
  </table>
  <div style="font-size: 7.5pt; color: #64748b; font-style: italic;">
    *Note: The deployed {dep_label} ECE drops from {deployed.get('ece', 0):.4f} to {deployed.get('ece_after_temperature', 0):.4f} ({dep_ece_cal}) following post-hoc Temperature Scaling. The other rows show their uncalibrated ECE.
  </div>
</div>

<div class="grid-2 avoid-break">
  {"<div class='figure-box'><img class='figure-img' src='" + img_compare_acc + "' alt='Accuracy Comparison'><div class='figure-caption'>Figure 3: Test Accuracy Comparison across Model Families.</div></div>" if img_compare_acc else ""}
  {"<div class='figure-box'><img class='figure-img' src='" + img_tradeoff + "' alt='Model Size vs Accuracy'><div class='figure-caption'>Figure 4: Pareto Frontier: Model Size (MB) vs. Test Accuracy.</div></div>" if img_tradeoff else ""}
</div>

<div class="page-break"></div>

<!-- ========================================================================= -->
<!-- 6. ATTENTION ABLATION & SCIENTIFIC NOISE FLOOR                           -->
<!-- ========================================================================= -->
<h2>6. Attention Ablation Study & Rigorous Analysis</h2>

<p>
A critical hallmark of scientific integrity is verifying whether improvements attributed to architectural novelty (such as attention mechanisms) exceed the background noise floor of training non-determinism. We executed a controlled 5-arm ablation study on the Custom CNN architecture, keeping all data, seeds, optimizer states, and budgets identical.
</p>

<div class="avoid-break">
  <h3>Controlled Attention Ablation Arms</h3>
  <table>
    <thead>
      <tr>
        <th>Arm</th>
        <th>Architecture Configuration</th>
        <th class="numeric">Accuracy</th>
        <th class="numeric">&Delta; Acc</th>
        <th class="numeric">Macro F1</th>
        <th class="numeric">&Delta; F1</th>
        <th class="numeric">ECE</th>
        <th class="numeric">Param Overhead</th>
        <th class="numeric">Latency Overhead</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>A</strong></td>
        <td>CNN Baseline (No Attention)</td>
        <td class="numeric">95.88%</td>
        <td class="numeric">0.00%</td>
        <td class="numeric">0.9582</td>
        <td class="numeric">0.0000</td>
        <td class="numeric">0.0677</td>
        <td class="numeric">0 (+0.0%)</td>
        <td class="numeric">1.92 ms (+0%)</td>
      </tr>
      <tr>
        <td><strong>B</strong></td>
        <td>CNN + Channel Attention Only</td>
        <td class="numeric">96.34%</td>
        <td class="numeric">+0.46%</td>
        <td class="numeric">0.9631</td>
        <td class="numeric">+0.0049</td>
        <td class="numeric">0.0429</td>
        <td class="numeric">+11,008 (+0.88%)</td>
        <td class="numeric">4.22 ms (+119%)</td>
      </tr>
      <tr>
        <td><strong>C</strong></td>
        <td>CNN + Spatial Attention Only</td>
        <td class="numeric">96.12%</td>
        <td class="numeric">+0.24%</td>
        <td class="numeric">0.9606</td>
        <td class="numeric">+0.0024</td>
        <td class="numeric">0.0657</td>
        <td class="numeric">+392 (+0.03%)</td>
        <td class="numeric">3.25 ms (+69%)</td>
      </tr>
      <tr>
        <td><strong>D</strong></td>
        <td>CNN + Squeeze-and-Excitation (SE)</td>
        <td class="numeric"><strong>96.38%</strong></td>
        <td class="numeric"><strong>+0.50%</strong></td>
        <td class="numeric"><strong>0.9634</strong></td>
        <td class="numeric"><strong>+0.0051</strong></td>
        <td class="numeric">0.0582</td>
        <td class="numeric">+11,008 (+0.88%)</td>
        <td class="numeric">5.37 ms (+180%)</td>
      </tr>
      <tr style="background-color: #f0fdf4;">
        <td><strong>E</strong></td>
        <td>CNN + Full CBAM (Channel &rarr; Spatial)</td>
        <td class="numeric">96.29%</td>
        <td class="numeric">+0.42%</td>
        <td class="numeric">0.9628</td>
        <td class="numeric">+0.0045</td>
        <td class="numeric"><strong>0.0413</strong></td>
        <td class="numeric">+11,400 (+0.91%)</td>
        <td class="numeric">6.66 ms (+247%)</td>
      </tr>
    </tbody>
  </table>
</div>

<div class="grid-2 avoid-break">
  {"<div class='figure-box'><img class='figure-img' src='" + img_ablation + "' alt='Ablation Arms'><div class='figure-caption'>Figure 5: Ablation Arms Accuracy and Macro F1 Gains.</div></div>" if img_ablation else ""}
  {"<div class='figure-box'><img class='figure-img' src='" + img_placement + "' alt='CBAM Placement'><div class='figure-caption'>Figure 6: Accuracy across CBAM Stage Placements.</div></div>" if img_placement else ""}
</div>

<div class="callout callout-amber avoid-break">
  <div class="callout-title">Scientific Honesty: Noise Floor Analysis & Verdict</div>
  <p>
  Retraining <code>cnn_baseline</code> across 3 independent random seeds exhibited an empirical standard deviation of <strong>0.11 points (0.22 range)</strong> in Macro F1. Furthermore, repeating identical configurations across differing DataLoader worker configurations (2 vs 4 workers) shifted Macro F1 by up to <strong>0.47 points</strong> due to multi-process batch-order variance.
  </p>
  <p>
  <strong>Official Verdict:</strong> While CBAM improved macro F1 by <strong>+0.45 points</strong> over the baseline and delivered the lowest Expected Calibration Error (0.0413), this gain lies near the 0.47-point empirical noise threshold. Therefore, the attention gain is <em>directionally positive and improves feature localization</em>, but cannot be claimed as a statistically bulletproof delta without hundreds of seed replicates. Squeeze-and-Excitation (SE) was marginally the highest scoring arm (+0.51 points F1).
  </p>
</div>

<!-- ========================================================================= -->
<!-- 7. CALIBRATION, OOD DETECTION & IMAGE QUALITY                            -->
<!-- ========================================================================= -->
<h2>7. Statistical Calibration, OOD Detection & Quality Gating</h2>

<p>
A deployable agricultural diagnostic system cannot afford overconfident hallucinations when presented with blurred photos, corrupt data, or non-plant subjects. We implemented a three-layer trust stack:
</p>

<div class="avoid-break">
  <h3>1. Post-Hoc Temperature Scaling</h3>
  <p>
  Modern deep neural networks with batch normalization and label smoothing are prone to miscalibration—stating 99% probability on samples where true accuracy is 85%. We applied <strong>Temperature Scaling</strong> (Guo et al., 2017) by fitting a single scalar parameter <i>T</i> on the held-out validation set to minimize negative log-likelihood:
  </p>
  <div style="background: #f8fafc; border: 1px solid #e2e8f0; padding: 6px 12px; border-radius: 6px; font-size: 8.5pt; text-align: center; margin: 6px 0;">
    <i>p&#770;<sub>i</sub></i> = exp(<i>z<sub>i</sub> / T</i>) / &sum;<sub>j</sub> exp(<i>z<sub>j</sub> / T</i>)
  </div>
  <p>
  For the deployed {dep_label} model, the optimal temperature was fitted at <strong><i>T</i> = {dep_t}</strong>. Because a scalar divisor does not change the argmax ranking of the logits, classification accuracy remains completely unchanged, but Expected Calibration Error (ECE) fell from <strong>{dep_ece_raw} down to {dep_ece_cal}</strong> (about a {dep_ece_gain}&times; calibration precision improvement).
  </p>
</div>

<div class="grid-2 avoid-break">
  {"<div class='figure-box'><img class='figure-img' src='" + img_reliability + "' alt='Reliability Diagram'><div class='figure-caption'>Figure 7: Reliability Diagram Before vs. After Temperature Scaling.</div></div>" if img_reliability else ""}
  {"<div class='figure-box'><img class='figure-img' src='" + img_cm + "' alt='Confusion Matrix'><div class='figure-caption'>Figure 8: Normalized Test Confusion Matrix (17,572 Images).</div></div>" if img_cm else ""}
</div>

<div class="avoid-break">
  <h3>2. Out-of-Distribution (OOD) Gating</h3>
  <p>
  To reject non-leaf images, arbitrary photographs, or severe anomalies, the server scores every forward pass against three calibrated metrics:
  </p>
  <ul>
    <li><strong>Maximum Softmax Probability (MSP):</strong> Rejects predictions where peak confidence &lt; 0.806.</li>
    <li><strong>Normalized Predictive Entropy:</strong> Flags diffuse uncertainty where entropy &gt; 0.270.</li>
    <li><strong>Free Energy Score:</strong> Rejects uncharacteristic logit energies exceeding -4.625.</li>
  </ul>
  <p>
  Thresholds are calibrated to maintain a strict <strong>95% True Positive Rate</strong> on genuine in-distribution test leaves, ensuring predictable real-world behavior.
  </p>
</div>

<div class="page-break"></div>

<!-- ========================================================================= -->
<!-- 8. COMPLETE TOOL & TECHNOLOGY STACK                                      -->
<!-- ========================================================================= -->
<h2>8. Complete Tool & Technology Stack</h2>

<p>
The system leverages a production-grade Python, Deep Learning, and TypeScript/React ecosystem:
</p>

<div class="avoid-break">
  <table>
    <thead>
      <tr>
        <th>Engineering Tier</th>
        <th>Technologies Used</th>
        <th>Version / Stack</th>
        <th>Role & Architecture Rationale</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>Core Language</strong></td>
        <td>Python</td>
        <td>3.12 / 3.13</td>
        <td>Modern runtime supporting PyTorch CUDA and ONNX Runtime.</td>
      </tr>
      <tr>
        <td><strong>Deep Learning</strong></td>
        <td>PyTorch, Torchvision</td>
        <td>2.13.0+cu126</td>
        <td>Model authoring, autograd, GPU-accelerated training, and weights management.</td>
      </tr>
      <tr>
        <td><strong>Serving Engine</strong></td>
        <td>ONNX Runtime</td>
        <td>&ge; 1.18.0</td>
        <td>Inference execution engine on CPU; eliminates PyTorch's 530MB resident memory.</td>
      </tr>
      <tr>
        <td><strong>Classical ML</strong></td>
        <td>Scikit-learn, XGBoost</td>
        <td>&ge; 1.4, &ge; 2.0</td>
        <td>Logistic Regression, Linear SVM, Random Forest, and XGBoost baselines.</td>
      </tr>
      <tr>
        <td><strong>Computer Vision</strong></td>
        <td>OpenCV (headless), Pillow</td>
        <td>cv2 &ge; 4.9, PIL &ge; 10.2</td>
        <td>Quality assessment, blur variance of Laplacian, and deterministic transforms.</td>
      </tr>
      <tr>
        <td><strong>Backend API</strong></td>
        <td>FastAPI, Uvicorn, Starlette</td>
        <td>FastAPI 0.115, Uvicorn 0.30</td>
        <td>Asynchronous ASGI web framework with OpenAPI/Swagger auto-generation.</td>
      </tr>
      <tr>
        <td><strong>Data Contracts</strong></td>
        <td>Pydantic v2, Pydantic-Settings</td>
        <td>&ge; 2.7, &ge; 2.3</td>
        <td>Strict runtime request/response validation and environment configuration.</td>
      </tr>
      <tr>
        <td><strong>Persistence</strong></td>
        <td>SQLite, SQLAlchemy</td>
        <td>SQLAlchemy 2.0 (WAL)</td>
        <td>Zero-admin embedded database storing diagnosis records and timing logs.</td>
      </tr>
      <tr>
        <td><strong>Frontend Core</strong></td>
        <td>React 19, Vite 8</td>
        <td>React 19.2, Vite 8.2</td>
        <td>Blazing-fast client dashboard with Hot Module Replacement and ES modules.</td>
      </tr>
      <tr>
        <td><strong>UI & Styling</strong></td>
        <td>Tailwind CSS v4, Lucide Icons</td>
        <td>Tailwind 4.3, Lucide 1.35</td>
        <td>Clean design system with full light/dark mode and responsive layouts.</td>
      </tr>
      <tr>
        <td><strong>Data Visualization</strong></td>
        <td>Recharts</td>
        <td>Recharts 3.10</td>
        <td>Declarative SVG charts for benchmarks, training curves, and scatter plots.</td>
      </tr>
      <tr>
        <td><strong>Containerization</strong></td>
        <td>Docker (Multi-stage)</td>
        <td>BuildKit / Engine 27+</td>
        <td>Multi-stage container: Stage 1 Node (Vite build) + Stage 2 Python slim.</td>
      </tr>
      <tr>
        <td><strong>Cloud Deployment</strong></td>
        <td>Render Cloud</td>
        <td>Blueprint (render.yaml)</td>
        <td>Single container orchestration for API + Frontend SPA on 512 MB Free Tier.</td>
      </tr>
      <tr>
        <td><strong>Test Automation</strong></td>
        <td>Pytest, HTTPX, AnyIO</td>
        <td>Pytest 8.4, Respx 0.23</td>
        <td>92 automated tests across API endpoints, deployment configs, and ONNX serving.</td>
      </tr>
    </tbody>
  </table>
</div>

<!-- ========================================================================= -->
<!-- 9. PRODUCTION DEPLOYMENT ON RENDER                                       -->
<!-- ========================================================================= -->
<h2>9. Production Deployment Engineering on Render</h2>

<div class="avoid-break">
  <h3>The 512 MB Memory Challenge & Solution</h3>
  <p>
  Render's Free Web Service tier enforces a hard <strong>512 MB physical RAM ceiling</strong>. Running the standard PyTorch serving stack requires roughly <strong>657 MB of resident memory</strong> before serving its first request (Python runtime + PyTorch C++ libraries + CUDA stubs + weights), causing Render's kernel OOM-killer to terminate the container during boot.
  </p>
  <p>
  We engineered a zero-loss architectural solution:
  </p>
  <ul>
    <li><strong>Torch-Free Serving Image (<code>requirements-serve.txt</code>):</strong> Completely eliminates <code>torch</code> and <code>torchvision</code> from the production image, replacing them with <code>onnxruntime</code>.</li>
    <li><strong>Bit-Exact Preprocessing:</strong> Reimplemented torchvision's evaluation pipeline using pure NumPy and Pillow (<code>app/ml/preprocess.py</code>), proven bit-exact with torchvision down to 2.2 &times; 10<sup>-6</sup> on class probabilities.</li>
    <li><strong>Massive Footprint Reduction:</strong> Drops container resident set size to <strong>~115 MB – 135 MB</strong>, providing over <strong>3.5&times; headroom</strong> within Render's 512 MB budget.</li>
    <li><strong>Glibc Memory Arena Capping:</strong> Added <code>MALLOC_ARENA_MAX=2</code> to prevent glibc from allocating up to 8 arenas per CPU core under concurrent request bursts.</li>
  </ul>
</div>

<div class="avoid-break">
  <h3>Single-Container Multi-Stage Docker Build</h3>
  <p>
  Rather than maintaining separate hosting bills and complex CORS configurations for frontend and backend, <code>docker/render.Dockerfile</code> packs both into one cohesive container:
  </p>
  <ul>
    <li><strong>Stage 1 (Node 22-slim):</strong> Compiles React 19 source into static HTML, CSS, and JS bundles via <code>npm run build</code>.</li>
    <li><strong>Stage 2 (Python 3.12-slim):</strong> Copies compiled frontend into <code>frontend/dist/</code>, installs <code>requirements-serve.txt</code>, bakes in <code>serving.onnx</code> (16.4 MB) and JSON metadata, creates a secure non-root user (<code>appuser:10001</code>), and binds Uvicorn to <code>$PORT</code>.</li>
    <li>FastAPI automatically detects <code>frontend/dist/index.html</code> at startup and serves the client app at <code>/</code> while routing <code>/api/*</code> to REST endpoints.</li>
  </ul>
</div>

<div class="page-break"></div>

<!-- ========================================================================= -->
<!-- 10. REAL-WORLD DOMAIN SHIFT, LIMITATIONS & ROADMAP                       -->
<!-- ========================================================================= -->
<h2>10. Real-World Domain Shift, Limitations & Future Roadmap</h2>

<div class="avoid-break">
  <h3>The Laboratory Bias & Domain Shift Probe</h3>
  <p>
  A critical engineering limitation of the PlantVillage dataset is that images were captured under controlled laboratory conditions (single detached leaves laid flat against plain gray or black backgrounds under studio lighting). To quantify how a model degrades when transitioning to real-world agricultural fields, a <strong>Domain Shift Probe</strong> (<code>scripts/domain_shift_probe.py</code>) applies six environmental transformations to {shift_n} held-out disease leaves per condition and sends each through the complete serving pipeline. The table below is for the <strong>deployed, field-trained model</strong>. A refusal produces no class and counts as not correct, which is why even the unmodified photographs score below the laboratory test accuracy: the pipeline's own checks refuse a few percent of clean laboratory photographs.
  </p>

  <table>
    <thead>
      <tr>
        <th>Environmental Condition</th>
        <th>Applied Transformation</th>
        <th>Correct class</th>
        <th>Not marked reliable</th>
      </tr>
    </thead>
    <tbody>
      {shift_rows}
    </tbody>
  </table>
</div>

<div class="avoid-break">
  <h3>Closing the Gap: Field-Condition Training</h3>
  <p>
  The first version of the production model was trained on laboratory photographs only. Measured through the same pipeline it kept ~96% on clean photographs but fell to about 70% with all field effects combined and to under 10% when the leaf was small in the frame. Two further training rounds added <strong>field-condition augmentation</strong> (<code>backend/app/ml/field_augment.py</code>): leaf cut-outs placed on procedural soil, mulch, foliage and stone backdrops, small-in-frame placement, directional light and shadow, blur, sensor noise and JPEG compression. The leaf's hue is never altered, because colour is the diagnostic signal. Round one applied it to 70% of images with the original schedule; round two applied it to 50%, fine-tuned the <em>whole</em> network instead of the last three blocks, and trained for ten epochs. The table compares every model through the identical pipeline on the same {comparison_n} held-out leaves per condition, with the number of confident wrong answers in brackets:
  </p>
  <table>
    <thead>
      <tr>
        <th>Model</th>
        <th class="numeric">Lab test</th>
        <th class="numeric">Original</th>
        <th class="numeric">Background</th>
        <th class="numeric">Angle</th>
        <th class="numeric">Lighting</th>
        <th class="numeric">Compression</th>
        <th class="numeric">Distance</th>
        <th class="numeric">Combined</th>
      </tr>
    </thead>
    <tbody>
      {comparison_rows}
    </tbody>
  </table>
</div>

<div class="callout callout-amber avoid-break">
  <div class="callout-title">Reading These Results Honestly</div>
  <strong>Not independent.</strong> The augmentation imitates the same kinds of condition the probe applies (through different generators), so the probe flatters the field-trained models; real field photographs are the independent test, reported next. <strong>Not attributable to CBAM.</strong> In round one, plain EfficientNet-B0 and EfficientNet-B0 + CBAM tied (98.32% vs 98.25%). Round two trained only the CBAM model with three changes at once (longer schedule, whole network, milder augmentation), so the gain over the earlier models cannot be assigned to any single one of them, and no plain EfficientNet was run on that schedule. The round-two run was also interrupted by a machine restart during its final epoch; its best checkpoint (epoch 9 of 10) was evaluated exactly as a finished run would have been.
</div>

<div class="avoid-break">
  <h3>Accuracy on Real Photographs</h3>
  <p>
  Laboratory accuracy (~99%) does not carry over to a photograph from a phone or a web search. The independent test is the held-out test split of PlantDoc &mdash; {real_n} real field and web photographs that no model saw in training (perceptual-hash de-duplication removed any training photo matching a test photo). The models trained on laboratory photographs only are right on about a third of them; fine-tuning on eight real-photo datasets raises that to <strong>{real_top1}</strong>, with the right disease among the top three <strong>{real_top3}</strong> of the time:
  </p>
  <table>
    <thead><tr><th>Model</th><th>Trained on</th><th class="numeric">Real photos: top-1 (95% CI)</th><th class="numeric">Top-3</th><th class="numeric">Lab test</th></tr></thead>
    <tbody>
      {real_rows}
    </tbody>
  </table>
</div>

<div class="avoid-break">
  <h3>Real-Photo Training Data</h3>
  <p>Only classes that map unambiguously onto the model's 38 were used; photographs were resized to 512&nbsp;px and de-duplicated. A stratified PlantVillage subset was mixed in so no class is forgotten. No photographs are committed to the repository.</p>
  <table>
    <thead><tr><th>Source</th><th>Licence</th><th>Citation</th></tr></thead>
    <tbody>
      {source_rows}
    </tbody>
  </table>
  <div style="font-size: 7.5pt; color: #64748b; font-style: italic;">PlantWild (CC BY-NC-ND 4.0) and PlantSeg (CC BY-NC 4.0) are licensed for non-commercial use only and CD&amp;S states no licence, so the deployed model is for research and education.</div>
</div>

<div class="avoid-break">
  <h3>When the Application Says "Not Sure"</h3>
  <p>
  The threshold below which the app gives no plain diagnosis was chosen on {op_photos} real validation photographs (never the test split) as the most permissive value at which answers shown as a diagnosis are right at least 90% of the time overall and at least 85% on the photos found through web image searches (PlantDoc, PlantWild, PlantSeg) on their own &mdash; pooled with the easy field collections, a threshold could otherwise look precise while letting through confidently wrong answers on exactly the photos people check against a web search. At the previous setting (0.15) this model would give a confidently wrong diagnosis for {op_wrong_before} of real photos; at {dep_msp} that falls to {op_wrong_now}, and the uncertain photos are shown with their closest matches instead.
  </p>
  <table>
    <thead><tr><th class="numeric">Threshold</th><th class="numeric">Shown as diagnosis</th><th class="numeric">Of those, right</th><th class="numeric">Confidently wrong (of all)</th></tr></thead>
    <tbody>
      {op_rows}
    </tbody>
  </table>
</div>

<div class="avoid-break">
  <h3>A Bigger Model, and Both Together</h3>
  <p>ResNet50 + CBAM (92&nbsp;MB) was fine-tuned on five of the real-photo sources with the same schedule, and averaging its calibrated probabilities with the served model's was measured. Top-1 / top-3 on real photos:</p>
  <table>
    <thead><tr><th>Split</th><th class="numeric">EfficientNet-B0 + CBAM</th><th class="numeric">ResNet50 + CBAM</th><th class="numeric">Average of both</th></tr></thead>
    <tbody>
      {ensemble_rows}
    </tbody>
  </table>
  <p>Alone the two are within a point of each other; averaged they are the most accurate configuration measured on real photos &mdash; about three points better on the test photos and five on PlantWild's validation photos, at the top of the 60&ndash;76% range published for this benchmark. Serving the average would add the 92&nbsp;MB ResNet50 and several times the CPU time per photo on the 512&nbsp;MB free host, so the application serves the EfficientNet model; the ensemble is implemented end to end for a larger host.</p>
</div>

<div class="avoid-break">
  <h3>Future Roadmap for Production Agriculture</h3>
  <ul>
    <li><strong>Two-Stage Object Detection (YOLOv10 / RT-DETR):</strong> Incorporate an upstream bounding-box detector to crop individual leaves out of messy canopy photographs before classification.</li>
    <li><strong>In-Field Active Learning:</strong> Deploy mobile edge instances allowing farmers to flag inaccurate predictions and upload verified field images to periodically fine-tune the backbone.</li>
    <li><strong>Multi-Lingual Farmer Interface:</strong> Expand disease advisory content in Hindi, Spanish, Swahili, and regional agricultural dialects.</li>
    <li><strong>On-Device Mobile Deployment (TFLite / ONNX Mobile):</strong> Quantize the {dep_size} MB ONNX model to 8-bit integer weights (INT8 ~4 MB) to run directly inside offline native mobile applications.</li>
  </ul>
</div>

<!-- ========================================================================= -->
<!-- 11. CONCLUSION                                                           -->
<!-- ========================================================================= -->
<h2>11. Conclusion & Verification Summary</h2>

<p>
This project demonstrates that deploying deep learning vision models for agricultural disease diagnosis requires far more than training a high-accuracy classifier. By implementing:
</p>
<ol>
  <li><strong>Controlled Scientific Rigor:</strong> 18 benchmark architectures, ablation study with noise-floor benchmarking, and 3-way split protocol.</li>
  <li><strong>Measured Model Selection:</strong> {dep_label} achieving <strong>{dep_acc} accuracy, {dep_f1_pct} Macro F1</strong> with small size ({dep_size} MB) and fast CPU inference (~{inference_ms} ms), fine-tuned on eight real-photo datasets: on held-out real photographs it is right {real_top1} of the time (top-3 {real_top3}), up from {old_real_top1} for the laboratory-trained model.</li>
  <li><strong>Production Safety & Trust:</strong> Image quality filtering, a dust/soil/sand detector, out-of-distribution rejection, and post-hoc temperature scaling bringing Expected Calibration Error down to <strong>{dep_ece_cal}</strong>.</li>
  <li><strong>Cloud Engineering Excellence:</strong> A single-container multi-stage architecture powered by ONNX Runtime, effortlessly fitting into Render's 512 MB free tier with 4&times; safety headroom.</li>
</ol>
<p>
The complete codebase, configurations, Dockerfiles, and test suites are verified and ready for production operation.
</p>

</body>
</html>
"""

    html_path = PROJECT_ROOT / "Plant_Disease_Detection_Project_Report.html"
    pdf_path = PROJECT_ROOT / "Plant_Disease_Detection_Project_Report.pdf"

    print(f"Writing HTML report to {html_path}...")
    html_path.write_text(html_content, encoding="utf-8")

    print("Launching Playwright to render PDF...")
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.set_content(html_content, wait_until="networkidle")
        
        page.pdf(
            path=str(pdf_path),
            format="A4",
            margin={
                "top": "20mm",
                "bottom": "20mm",
                "left": "16mm",
                "right": "16mm",
            },
            print_background=True,
            display_header_footer=True,
            header_template='<div style="font-family: -apple-system, sans-serif; font-size: 7.5pt; color: #94a3b8; width: 100%; text-align: right; padding-right: 18mm;">Plant Disease Detection System — Technical & Research Report</div>',
            footer_template='<div style="font-family: -apple-system, sans-serif; font-size: 7.5pt; color: #94a3b8; width: 100%; display: flex; justify-content: space-between; padding: 0 18mm;"><span>AtharvDhiman/plant-disease-detection</span><span>Page <span class="pageNumber"></span> of <span class="totalPages"></span></span></div>',
        )
        browser.close()

    print(f"SUCCESS! PDF report successfully generated at: {pdf_path}")
    print(f"PDF size: {pdf_path.stat().st_size / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    generate_report()
