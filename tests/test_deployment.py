"""Static checks on the deployment configuration.

Docker is not installed on the development machine, so the images could not be
built and run here. These tests check what *can* be checked without a Docker
daemon: that the compose file parses, that the services and volumes referenced
actually line up, that every Dockerfile uses only real instructions, and that
the paths the compose file mounts exist in the repository.

That is a real, if partial, guarantee - and the README says plainly which parts
were verified by execution and which were not.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DOCKERFILE_DIRECTIVES = {
    "FROM", "RUN", "CMD", "LABEL", "EXPOSE", "ENV", "ADD", "COPY", "ENTRYPOINT",
    "VOLUME", "USER", "WORKDIR", "ARG", "ONBUILD", "STOPSIGNAL", "HEALTHCHECK", "SHELL",
}


def _instructions(path: Path) -> list[tuple[int, str, str]]:
    """Yield ``(line number, directive, remainder)`` for each instruction."""
    found: list[tuple[int, str, str]] = []
    continuing = False
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if continuing:
            continuing = line.endswith("\\")
            continue
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        found.append((number, parts[0].upper(), parts[1] if len(parts) > 1 else ""))
        continuing = line.endswith("\\")
    return found


@pytest.fixture(scope="module")
def compose() -> dict:
    yaml = pytest.importorskip("yaml", reason="pyyaml not installed")
    return yaml.safe_load((PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))


def test_compose_file_parses(compose):
    assert "services" in compose
    assert set(compose["services"]) == {"backend", "frontend"}


def test_compose_dockerfiles_exist(compose):
    for name, service in compose["services"].items():
        dockerfile = PROJECT_ROOT / service["build"]["dockerfile"]
        assert dockerfile.is_file(), f"{name} references a missing {dockerfile}"


def test_compose_named_volumes_are_declared(compose):
    declared = set(compose.get("volumes") or {})
    for name, service in compose["services"].items():
        for mount in service.get("volumes", []):
            source = mount.split(":", 1)[0]
            if source.startswith((".", "/")):
                continue  # bind mount, checked separately
            assert source in declared, f"{name} mounts undeclared volume {source!r}"


def test_compose_bind_mounts_exist(compose):
    for name, service in compose["services"].items():
        for mount in service.get("volumes", []):
            source = mount.split(":", 1)[0]
            if not source.startswith("."):
                continue
            path = PROJECT_ROOT / source
            assert path.exists(), f"{name} bind-mounts a missing path: {source}"


def test_model_artifacts_are_mounted_read_only(compose):
    """The API must never be able to modify a trained model or an artefact."""
    mounts = compose["services"]["backend"]["volumes"]
    for protected in ("./models:", "./artifacts:"):
        entry = next((m for m in mounts if m.startswith(protected)), None)
        assert entry is not None, f"{protected} is not mounted"
        assert entry.endswith(":ro"), f"{entry} should be read-only"


def test_frontend_publishes_a_port_and_backend_does_not(compose):
    """Only nginx should be reachable; the API is proxied behind it."""
    assert compose["services"]["frontend"].get("ports")
    assert not compose["services"]["backend"].get("ports")


# Every Dockerfile in docker/, not a hand-maintained list: the cloudrun and
# render images cannot be built on the development machine, so this static parse
# is the only check they get and it must not silently skip them.
@pytest.mark.parametrize(
    "name",
    sorted(p.name for p in (PROJECT_ROOT / "docker").glob("*.Dockerfile")),
)
def test_dockerfile_instructions_are_valid(name):
    path = PROJECT_ROOT / "docker" / name
    assert path.is_file()
    instructions = _instructions(path)
    assert instructions, "Dockerfile has no instructions"
    for number, directive, _ in instructions:
        assert directive in DOCKERFILE_DIRECTIVES, \
            f"{name}:{number} unknown instruction {directive!r}"
    assert instructions[0][1] in {"FROM", "ARG"}, "A Dockerfile must open with FROM or ARG"
    assert any(d == "FROM" for _, d, _ in instructions)
    assert any(d in {"CMD", "ENTRYPOINT"} for _, d, _ in instructions)


def test_backend_image_runs_as_non_root():
    directives = _instructions(PROJECT_ROOT / "docker" / "backend.Dockerfile")
    users = [rest.strip() for _, directive, rest in directives if directive == "USER"]
    assert users, "backend.Dockerfile never drops privileges with USER"
    assert users[-1] != "root"


def test_frontend_is_a_multi_stage_build():
    """The node toolchain must not ship in the runtime image."""
    directives = _instructions(PROJECT_ROOT / "docker" / "frontend.Dockerfile")
    froms = [rest for _, directive, rest in directives if directive == "FROM"]
    assert len(froms) >= 2, "frontend.Dockerfile should build then copy into a slim runtime"
    assert "nginx" in froms[-1].lower()


def test_nginx_proxies_api_to_the_backend_service():
    config = (PROJECT_ROOT / "docker" / "nginx.conf").read_text(encoding="utf-8")
    assert "proxy_pass http://backend:8000" in config
    # A single-page app must fall through to index.html on a hard refresh.
    assert "try_files" in config and "index.html" in config


def test_dockerignore_excludes_the_dataset_and_checkpoints():
    """A 3 GB build context would make every image build unusably slow."""
    ignored = (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8")
    for pattern in ("_kagglehub_cache/", "models/checkpoints/", "node_modules/"):
        assert pattern in ignored, f".dockerignore is missing {pattern}"


def test_env_example_documents_every_setting():
    """Every configurable field should appear in .env.example."""
    import sys

    sys.path.insert(0, str(PROJECT_ROOT / "backend"))
    from app.core.config import Settings

    example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    # Purely derived or path-internal fields do not need documenting.
    skip = {"project_root", "data_dir", "models_dir", "artifacts_dir", "api_title",
            "api_version", "normalize_mean", "normalize_std", "allowed_content_types"}
    missing = [
        name for name in Settings.model_fields
        if name not in skip and f"PDD_{name.upper()}" not in example
    ]
    assert not missing, f".env.example does not mention: {missing}"


def test_requirements_pin_the_critical_packages():
    text = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
    for package in ("torch", "torchvision", "fastapi", "SQLAlchemy", "scikit-learn",
                    "opencv-python-headless", "pytest"):
        assert re.search(rf"^{re.escape(package)}\b", text, re.MULTILINE), \
            f"requirements.txt does not list {package}"


# --------------------------------------------------------------------------- #
# Dataset resolution
#
# The pipeline must run on a machine that already has the dataset, with no
# Kaggle account and no network. These tests pin that: a valid local directory
# is accepted without kagglehub ever being imported, and an invalid one fails
# loudly instead of falling through to a download.
# --------------------------------------------------------------------------- #

def _make_dataset(root, class_count=6, nested=0):
    """Create a minimal ImageFolder tree, optionally nested `nested` levels deep."""
    target = root
    for level in range(nested):
        target = target / f"wrapper{level}"
    target = target / "train"
    for index in range(class_count):
        class_dir = target / f"Class___{index}"
        class_dir.mkdir(parents=True, exist_ok=True)
        (class_dir / "leaf.jpg").write_bytes(b"\xff\xd8\xff\xe0not-a-real-jpeg")
    return target


def test_flat_dataset_is_accepted(tmp_path):
    from scripts.download_dataset import validate_dataset_root

    _make_dataset(tmp_path)
    ok, detail = validate_dataset_root(tmp_path)
    assert ok is True
    assert "6 class folders" in detail


def test_kaggle_style_nesting_is_found(tmp_path):
    """The real archive buries the class folders three levels down."""
    from scripts.download_dataset import validate_dataset_root

    _make_dataset(tmp_path, nested=2)
    ok, _ = validate_dataset_root(tmp_path)
    assert ok is True


def test_directory_without_class_folders_is_rejected(tmp_path):
    from scripts.download_dataset import validate_dataset_root

    (tmp_path / "loose.jpg").write_bytes(b"\xff\xd8\xff\xe0")
    ok, detail = validate_dataset_root(tmp_path)
    assert ok is False
    assert "no directory with at least" in detail


def test_missing_directory_is_rejected(tmp_path):
    from scripts.download_dataset import validate_dataset_root

    ok, detail = validate_dataset_root(tmp_path / "nope")
    assert ok is False
    assert "does not exist" in detail


def test_local_dataset_resolves_without_importing_kagglehub(tmp_path, monkeypatch):
    """The offline path must never reach the module that needs credentials."""
    import builtins

    import scripts.download_dataset as dl

    _make_dataset(tmp_path)
    monkeypatch.setattr(dl, "RECORD_PATH", tmp_path / "record.json")

    real_import = builtins.__import__

    def guard(name, *args, **kwargs):
        if name == "kagglehub":
            raise AssertionError("kagglehub was imported on the offline path")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    monkeypatch.setattr(sys, "argv", ["download_dataset.py", "--path", str(tmp_path)])

    assert dl.main() == 0
    record = json.loads((tmp_path / "record.json").read_text(encoding="utf-8"))
    assert record["source"] == "--path"
    assert record["path"] == str(tmp_path)


def test_explicit_bad_path_stops_rather_than_downloading(tmp_path, monkeypatch):
    """An explicit --path that is wrong is a user error, not a reason to fetch."""
    import scripts.download_dataset as dl

    monkeypatch.setattr(sys, "argv", ["download_dataset.py", "--path", str(tmp_path / "absent")])
    with pytest.raises(SystemExit) as excinfo:
        dl.main()
    assert "not a usable dataset" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# Generated documentation
# --------------------------------------------------------------------------- #

def test_seed_replicates_are_excluded_from_generated_tables():
    """Replicates measure noise; they are not competitors in the comparison.

    They share a model name with the canonical run, so leaving them in printed
    the same architecture three times in the benchmark table, reading as three
    different models that happen to have identical names and sizes.
    """
    from scripts.generate_docs import rows_from_ledger

    def record(name, seed, accuracy):
        return {
            "label": name, "model_name": name, "group": "benchmark",
            "protocol": "benchmark", "proposed": False,
            "config": {"seed": seed, "image_size": 160},
            "total_parameters": 1_000_000, "model_size_mb": 4.8,
            "training_time_s": 600, "epochs_run": 12, "best_val_accuracy": accuracy,
            "test": {"metrics": {"accuracy": accuracy, "f1_macro": accuracy}},
        }

    ledger = [
        record("cnn_baseline", 42, 0.9583),
        record("cnn_baseline", 43, 0.9561),
        record("cnn_baseline", 44, 0.9575),
        record("resnet50", 42, 0.9912),
    ]
    rows = rows_from_ledger(ledger, ("benchmark",))

    names = [r["model_name"] for r in rows]
    assert names.count("cnn_baseline") == 1
    assert "resnet50" in names
    # The canonical run is the one kept, not whichever scored best.
    kept = next(r for r in rows if r["model_name"] == "cnn_baseline")
    assert kept["accuracy"] == 0.9583


def test_ledger_without_seeds_is_not_filtered_away():
    """An older ledger with no recorded seed must still produce rows."""
    from scripts.generate_docs import rows_from_ledger

    ledger = [{
        "label": "cnn_baseline", "model_name": "cnn_baseline", "group": "benchmark",
        "protocol": "benchmark", "proposed": False, "config": {},
        "total_parameters": 1_000_000, "model_size_mb": 4.8,
        "test": {"metrics": {"accuracy": 0.95, "f1_macro": 0.95}},
    }]
    assert len(rows_from_ledger(ledger, ("benchmark",))) == 1


# --------------------------------------------------------------------------- #
# Worker budgeting
# --------------------------------------------------------------------------- #

def _profile(**overrides):
    from config import HardwareProfile

    base = {
        "device": "cuda", "gpu_name": "GTX 1650", "vram_gb": 4.0,
        "system_ram_gb": 7.35, "cpu_count": 12, "cpu_name": "test",
        "supports_amp": False, "torch_version": "2.13.0", "cuda_version": "12.6",
    }
    base.update(overrides)
    return HardwareProfile(**base)


def test_workers_follow_commit_headroom_not_physical_ram():
    """The constraint is commit charge, so the budget must track it.

    A spawned worker reserves ~2 GB of commit charge re-importing the CUDA build
    of torch while its resident set stays small. Budgeting against physical RAM
    capped this machine at one worker even after its pagefile was enlarged
    enough to afford several.
    """
    from config import recommended_workers

    starved = _profile(free_commit_gb=8.0)
    roomy = _profile(free_commit_gb=17.5)

    assert recommended_workers(starved) < recommended_workers(roomy)
    assert recommended_workers(roomy) >= 4
    # RAM is unchanged between the two; only the pagefile-backed headroom moved.
    assert starved.system_ram_gb == roomy.system_ram_gb


def test_unmeasurable_commit_falls_back_to_the_conservative_estimate():
    """A failed probe must under-commit, never over-commit."""
    from config import recommended_workers

    unknown = _profile(free_commit_gb=0.0)
    assert recommended_workers(unknown) == 1  # (7.35 - 4) // 2


def test_worker_count_is_capped_by_cpu_count():
    from config import recommended_workers

    assert recommended_workers(_profile(free_commit_gb=64.0, cpu_count=4)) <= 2


def test_explicit_override_still_wins():
    from config import recommended_workers

    assert recommended_workers(_profile(free_commit_gb=2.0), override=3) == 3
    assert recommended_workers(_profile(free_commit_gb=64.0), override=0) == 0


def test_tuning_trials_are_persisted_as_they_complete():
    """A search that cannot resume is a search that may never finish here.

    The trial loop reads the ledger to skip completed trials, but nothing wrote
    to it until all six had finished - so the check could never fire. When the
    process was killed during trial 4, three completed trials and 46 minutes of
    GPU time were lost and the restart began again at trial 1.
    """
    source = (PROJECT_ROOT / "training" / "tune_hyperparameters.py").read_text(encoding="utf-8")

    # The write must happen inside the loop, before the summary is assembled.
    loop_body = source[source.index("for index, params in") if "for index, params in" in source
                       else 0: source.index("if not results:")]
    assert "tracker.log(" in loop_body, "trials are not written to the ledger inside the loop"
    assert "tracker.flush()" in loop_body, "the ledger is not flushed after each trial"

    # And the skip check must still be there for the write to be useful.
    assert "tracker.has(config.experiment_id)" in loop_body


# --------------------------------------------------------------------------- #
# Adding a dataset
# --------------------------------------------------------------------------- #

def test_find_class_root_handles_nested_layouts(tmp_path):
    """Datasets are packaged inconsistently; the class folders must be found.

    The Kaggle archive repeats its own name twice and hides the class folders
    under train/. Requiring a fixed layout would reject most real datasets.
    """
    from scripts.add_dataset import find_class_root

    nested = tmp_path / "Wheat Dataset" / "Wheat Dataset" / "train"
    for name in ("Wheat___Leaf_rust", "Wheat___healthy", "Wheat___Septoria"):
        (nested / name).mkdir(parents=True)
        (nested / name / "a.jpg").write_bytes(b"\xff\xd8\xff\xe0")

    assert find_class_root(tmp_path) == nested


def test_kaggle_credentials_check_never_reads_the_token(tmp_path, monkeypatch):
    """Presence is all that matters; the secret itself must never be opened."""
    from scripts.add_dataset import kaggle_credentials_present

    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    ok, detail = kaggle_credentials_present()
    assert ok is False
    assert "Create New Token" in detail

    token = tmp_path / ".kaggle" / "kaggle.json"
    token.parent.mkdir(parents=True)
    token.write_text('{"username":"u","key":"SECRET_VALUE"}', encoding="utf-8")

    ok, detail = kaggle_credentials_present()
    assert ok is True
    # The path may be reported; the key must never be.
    assert "SECRET_VALUE" not in detail


def test_environment_credentials_are_accepted(monkeypatch, tmp_path):
    from scripts.add_dataset import kaggle_credentials_present

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setenv("KAGGLE_USERNAME", "someone")
    monkeypatch.setenv("KAGGLE_KEY", "SECRET_VALUE")

    ok, detail = kaggle_credentials_present()
    assert ok is True
    assert "SECRET_VALUE" not in detail
