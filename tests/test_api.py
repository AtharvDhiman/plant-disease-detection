"""Integration tests for the HTTP API.

These exercise the real FastAPI app against the real exported model. Tests that
need a trained model skip cleanly when none has been exported, so the suite is
useful both before and after training.
"""
from __future__ import annotations

import pytest

# --------------------------------------------------------------------------- #
# System
# --------------------------------------------------------------------------- #


def test_health_reports_status(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "model_ready" in body
    assert body["confidence_thresholds"]["high"] > body["confidence_thresholds"]["medium"]


def test_health_sets_request_id_header(client):
    response = client.get("/api/health")
    assert response.headers["X-Request-ID"]
    assert float(response.headers["X-Response-Time-ms"]) >= 0


def test_openapi_schema_is_generated(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    for endpoint in ("/api/predict", "/api/predictions", "/api/models",
                     "/api/metrics", "/api/dataset-stats", "/api/diseases"):
        assert endpoint in paths


def test_unknown_route_returns_error_envelope(client):
    response = client.get("/api/does-not-exist")
    assert response.status_code == 404
    assert "error" in response.json()


# --------------------------------------------------------------------------- #
# Upload validation
# --------------------------------------------------------------------------- #


def test_rejects_non_image_upload(client, not_an_image):
    response = client.post(
        "/api/predict", files={"file": ("notes.txt", not_an_image, "text/plain")}
    )
    assert response.status_code in (400, 415)
    assert response.json()["error"]["code"] in {"unsupported_content_type", "not_an_image"}


def test_rejects_oversized_upload(client, oversized_bytes):
    response = client.post(
        "/api/predict", files={"file": ("huge.jpg", oversized_bytes, "image/jpeg")}
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "file_too_large"


def test_rejects_disallowed_extension(client, real_leaf_bytes):
    response = client.post(
        "/api/predict", files={"file": ("leaf.gif", real_leaf_bytes, "image/gif")}
    )
    assert response.status_code == 415


def test_rejects_empty_upload(client):
    response = client.post("/api/predict", files={"file": ("empty.jpg", b"", "image/jpeg")})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "empty_file"


def test_missing_file_is_a_validation_error(client):
    response = client.post("/api/predict")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


# --------------------------------------------------------------------------- #
# Image analysis (no model needed)
# --------------------------------------------------------------------------- #


def test_analyze_image_accepts_a_good_photo(client, real_leaf_bytes):
    response = client.post(
        "/api/analyze-image", files={"file": ("leaf.jpg", real_leaf_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["quality"]["score"] <= 1.0
    assert body["quality"]["band"] in {"good", "acceptable", "poor"}
    assert body["image"]["width"] > 0


def test_analyze_image_flags_blur(client, blurred_bytes):
    response = client.post(
        "/api/analyze-image", files={"file": ("blur.jpg", blurred_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    body = response.json()
    codes = {issue["code"] for issue in body["quality"]["issues"]}
    assert "blurry" in codes
    assert body["quality"]["passed"] is False


def test_analyze_image_flags_darkness(client, dark_bytes):
    response = client.post(
        "/api/analyze-image", files={"file": ("dark.jpg", dark_bytes, "image/jpeg")}
    )
    codes = {issue["code"] for issue in response.json()["quality"]["issues"]}
    assert "too_dark" in codes


def test_analyze_image_flags_tiny_images(client, tiny_bytes):
    response = client.post(
        "/api/analyze-image", files={"file": ("tiny.jpg", tiny_bytes, "image/jpeg")}
    )
    codes = {issue["code"] for issue in response.json()["quality"]["issues"]}
    assert "too_small" in codes


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #


@pytest.fixture
def prediction(client, model_available, real_leaf_bytes):
    if not model_available:
        pytest.skip("No exported model available")
    response = client.post(
        "/api/predict?top_k=5&explain=true",
        files={"file": ("leaf.jpg", real_leaf_bytes, "image/jpeg")},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_prediction_response_shape(prediction):
    for key in ("predicted_class", "display_name", "plant", "condition", "confidence",
                "confidence_level", "status", "top_predictions", "quality", "ood",
                "explanation", "model_info", "timings"):
        assert key in prediction, f"missing {key}"


def test_prediction_probabilities_are_valid(prediction):
    probabilities = [p["probability"] for p in prediction["top_predictions"]]
    assert probabilities == sorted(probabilities, reverse=True)
    assert all(0.0 <= p <= 1.0 for p in probabilities)
    assert abs(prediction["confidence"] - probabilities[0]) < 1e-6
    assert prediction["confidence_level"] in {"high", "medium", "low"}


def test_prediction_status_is_known(prediction):
    assert prediction["status"] in {"ok", "low_confidence", "out_of_distribution", "poor_quality"}


def test_prediction_includes_timings(prediction):
    timings = prediction["timings"]
    assert timings["inference_ms"] > 0
    assert timings["total_ms"] >= timings["inference_ms"]


def test_prediction_generates_explanations(prediction):
    explanation = prediction["explanation"]
    assert "grad_cam" in explanation["images"]
    assert "original" in explanation["images"]
    assert explanation["images"]["grad_cam"].startswith("/api/images/")


def test_explanation_images_are_served(client, prediction):
    for url in prediction["explanation"]["images"].values():
        response = client.get(url)
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("image/")


def test_prediction_includes_disease_info(prediction):
    info = prediction["disease_info"]
    assert info is not None
    assert info["class_name"] == prediction["predicted_class"]
    assert info["disclaimer"]


def test_prediction_can_skip_explanations(client, model_available, real_leaf_bytes):
    if not model_available:
        pytest.skip("No exported model available")
    response = client.post(
        "/api/predict?explain=false&save=false",
        files={"file": ("leaf.jpg", real_leaf_bytes, "image/jpeg")},
    )
    body = response.json()
    assert body["id"] is None
    assert body["explanation"]["available"] == []
    assert body["timings"]["explain_ms"] == 0


def test_poor_quality_image_is_reported_not_classified(client, model_available, dark_bytes):
    if not model_available:
        pytest.skip("No exported model available")
    response = client.post(
        "/api/predict", files={"file": ("dark.jpg", dark_bytes, "image/jpeg")}
    )
    body = response.json()
    assert body["status"] == "poor_quality"
    assert body["reliable"] is False
    assert "insufficient" in body["status_message"].lower()


# --------------------------------------------------------------------------- #
# History
# --------------------------------------------------------------------------- #


def test_history_lists_the_saved_prediction(client, prediction):
    response = client.get("/api/predictions?limit=5")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] >= 1
    assert any(item["id"] == prediction["id"] for item in body["items"])


def test_history_detail(client, prediction):
    response = client.get(f"/api/predictions/{prediction['id']}")
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] == prediction["predicted_class"]
    assert body["disease_info"]["class_name"] == prediction["predicted_class"]


def test_history_filtering_and_sorting(client, prediction):
    filtered = client.get(f"/api/predictions?disease={prediction['predicted_class']}")
    assert filtered.status_code == 200
    assert all(i["predicted_class"] == prediction["predicted_class"]
               for i in filtered.json()["items"])

    sorted_response = client.get("/api/predictions?sort=confidence_desc&limit=10")
    confidences = [i["confidence"] for i in sorted_response.json()["items"]]
    assert confidences == sorted(confidences, reverse=True)


def test_history_search(client, prediction):
    needle = prediction["plant"][:4]
    response = client.get(f"/api/predictions?search={needle}")
    assert response.status_code == 200


def test_missing_prediction_returns_404(client):
    assert client.get("/api/predictions/999999").status_code == 404


def test_clear_history_requires_confirmation(client):
    assert client.delete("/api/predictions").status_code == 400


def test_delete_prediction(client, model_available, real_leaf_bytes):
    if not model_available:
        pytest.skip("No exported model available")
    created = client.post(
        "/api/predict", files={"file": ("leaf.jpg", real_leaf_bytes, "image/jpeg")}
    ).json()
    response = client.delete(f"/api/predictions/{created['id']}")
    assert response.status_code == 200
    assert response.json()["deleted"] == 1
    assert client.get(f"/api/predictions/{created['id']}").status_code == 404


# --------------------------------------------------------------------------- #
# Research endpoints
# --------------------------------------------------------------------------- #


def test_models_endpoint(client):
    body = client.get("/api/models").json()
    assert "production" in body
    assert "available" in body


def test_metrics_endpoint(client):
    body = client.get("/api/metrics").json()
    assert "rows" in body
    for row in body["rows"]:
        assert row["accuracy"] is None or 0.0 <= row["accuracy"] <= 1.0


def test_dataset_stats(client):
    response = client.get("/api/dataset-stats")
    if response.status_code == 404:
        pytest.skip("Dataset report not generated")
    body = response.json()
    assert body["num_classes"] > 0
    assert body["total_images"] > 0
    assert "imbalance" in body


def test_disease_library(client):
    body = client.get("/api/diseases").json()
    assert body["count"] > 0
    assert body["plants"]
    entry = body["items"][0]
    for key in ("class_name", "display_name", "plant", "category", "severity"):
        assert key in entry


def test_disease_detail(client):
    listing = client.get("/api/diseases").json()
    class_name = listing["items"][0]["class_name"]
    body = client.get(f"/api/diseases/{class_name}").json()
    assert body["class_name"] == class_name
    assert body["disclaimer"]


def test_disease_filters(client):
    body = client.get("/api/diseases?category=fungal").json()
    assert all(i["category"] == "fungal" for i in body["items"])


def test_dashboard(client):
    body = client.get("/api/dashboard").json()
    assert "totals" in body
    assert body["totals"]["predictions"] >= 0
    assert len(body["confidence_histogram"]) > 0


def test_figures_listing(client):
    body = client.get("/api/figures").json()
    assert isinstance(body, dict)


def test_model_health(client, model_available):
    response = client.get("/api/health/model")
    if not model_available:
        assert response.status_code == 503
        return
    body = response.json()
    assert body["status"] == "ready"
    assert body["num_classes"] > 0
    assert body["image_size"] > 0


# --------------------------------------------------------------------------- #
# Concurrency
# --------------------------------------------------------------------------- #


def test_concurrent_predictions_are_consistent(client, model_available, real_leaf_bytes):
    """The same image must give the same answer under concurrent load.

    FastAPI runs synchronous endpoints in a threadpool, so these requests really
    do overlap on one shared model. Grad-CAM mutates that model - hooks,
    requires_grad flags, accumulated gradients, cached attention gates - so
    without the registry's per-model lock the responses would interfere.
    """
    if not model_available:
        pytest.skip("No exported model available")

    from concurrent.futures import ThreadPoolExecutor

    def analyse():
        response = client.post(
            "/api/predict?explain=true&save=false",
            files={"file": ("leaf.jpg", real_leaf_bytes, "image/jpeg")},
        )
        assert response.status_code == 200, response.text
        return response.json()

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda _: analyse(), range(6)))

    classes = {r["predicted_class"] for r in results}
    assert len(classes) == 1, f"Concurrent requests disagreed: {classes}"

    confidences = {round(r["confidence"], 6) for r in results}
    assert len(confidences) == 1, f"Confidence varied under concurrency: {confidences}"

    for result in results:
        assert result["explanation"]["images"].get("grad_cam")
        stats = result["explanation"]["stats"]["grad_cam"]
        # A corrupted heat map shows up as an all-zero or saturated map.
        assert 0.0 < stats["peak_value"] <= 1.0


def test_explanation_methods_are_selectable(client, model_available, real_leaf_bytes):
    """Integrated gradients is available but opt-in, since it costs ~24 backward passes."""
    if not model_available:
        pytest.skip("No exported model available")

    default = client.post(
        "/api/predict?explain=true&save=false",
        files={"file": ("leaf.jpg", real_leaf_bytes, "image/jpeg")},
    ).json()
    assert "integrated_gradients" not in default["explanation"]["available"]

    requested = client.post(
        "/api/predict?explain=true&save=false"
        "&methods=grad_cam&methods=integrated_gradients",
        files={"file": ("leaf.jpg", real_leaf_bytes, "image/jpeg")},
    )
    assert requested.status_code == 200, requested.text
    available = requested.json()["explanation"]["available"]
    assert "integrated_gradients" in available
    assert "grad_cam" in available
    assert "grad_cam_plus_plus" not in available


def test_unknown_explanation_method_is_rejected(client, real_leaf_bytes):
    response = client.post(
        "/api/predict?methods=not_a_method",
        files={"file": ("leaf.jpg", real_leaf_bytes, "image/jpeg")},
    )
    assert response.status_code == 422


# --------------------------------------------------------------------------- #
# Research endpoints
#
# The dashboard grades how confident the ablation panel looks from fields the
# analysis computes. If the API stops forwarding them the UI silently falls back
# to a neutral tone, which would understate or overstate the result without any
# visible failure - so the contract is pinned here.
# --------------------------------------------------------------------------- #

def test_ablation_endpoint_forwards_evidence_grading(client):
    response = client.get("/api/metrics/ablation")
    assert response.status_code == 200
    payload = response.json()

    verdict = (payload.get("summary") or {}).get("verdict")
    if verdict is None:
        pytest.skip("ablation has not been run in this environment")

    assert verdict["evidence_strength"] in {
        "supported", "suggestive", "within noise", "no improvement",
    }
    assert verdict["significance_threshold_f1"] > 0
    # The UI renders this as a multiple of the floor; a null would print "NaNx".
    if verdict["cbam_delta_f1_macro"] > 0:
        assert verdict["margin_over_threshold"] is not None
        assert verdict["margin_over_threshold"] > 0


def test_ablation_threshold_is_measured_not_hardcoded(client):
    """The threshold must come from the ledger, not from a constant in the code.

    A hardcoded 0.005 would be the old rule-of-thumb; the measured floor is a
    different number, and the two disagreeing is exactly the drift this guards.
    """
    response = client.get("/api/metrics/ablation")
    summary = (response.json().get("summary") or {})
    verdict = summary.get("verdict")
    noise = summary.get("noise_floor") or {}
    if verdict is None or not noise.get("available"):
        pytest.skip("ablation or noise floor not available in this environment")

    assert verdict["significance_threshold_f1"] == noise["threshold_f1_macro"]


def test_benchmark_rows_exclude_seed_replicates(client):
    """One architecture must appear once per protocol, not once per seed.

    Seed replicates share a model name, parameter count and size with the
    canonical run. Listing them made the dashboard show the same model three
    times with three different scores, which reads as three architectures.
    """
    response = client.get("/api/metrics")
    assert response.status_code == 200
    rows = response.json().get("benchmark") or response.json().get("rows") or []
    if not rows:
        pytest.skip("no benchmark rows in this environment")

    seen = {}
    for row in rows:
        key = (row["model_name"], row.get("protocol"))
        seen[key] = seen.get(key, 0) + 1
    repeated = {k: v for k, v in seen.items() if v > 1}
    assert not repeated, f"model/protocol pairs listed more than once: {repeated}"


def test_placement_rows_exclude_production_reruns(client):
    """Three placements means three bars, not four.

    place_cbam_last2 was retrained under the production protocol, so including
    it labelled two bars "CBAM on last two stages" - the taller one being a
    training-budget difference, not a placement difference.
    """
    response = client.get("/api/metrics/ablation")
    assert response.status_code == 200
    placement = response.json().get("placement") or []
    if not placement:
        pytest.skip("placement suite not run in this environment")

    labels = [row["label"] for row in placement]
    assert len(labels) == len(set(labels)), f"duplicate placement labels: {labels}"
    assert all(row.get("protocol") != "production" for row in placement)
