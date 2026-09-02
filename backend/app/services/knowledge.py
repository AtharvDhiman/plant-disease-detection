"""Access layer for the disease knowledge base (``data/disease_info.json``).

Loaded once and cached. Every lookup degrades gracefully: an unknown class name
returns a derived placeholder rather than raising, so a model retrained with new
classes cannot take the API down before the knowledge base catches up.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)


def _pretty(raw: str) -> tuple[str, str, bool]:
    """Derive ``(plant, condition, is_healthy)`` from a raw class directory name."""
    if "___" in raw:
        plant, condition = raw.split("___", 1)
    else:
        plant, condition = raw, "unknown"
    plant = plant.replace("_", " ").strip()
    if "," in plant:
        head, tail = plant.split(",", 1)
        plant = f"{tail.strip().capitalize()} {head.strip().lower()}"
    condition = condition.replace("_", " ").strip()
    healthy = condition.lower().startswith("healthy")
    return plant, ("Healthy" if healthy else condition), healthy


class KnowledgeBase:
    """Lazy, thread-safe reader for the disease knowledge base."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or settings.disease_info_path
        self._data: dict | None = None
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- load
    def _ensure(self) -> dict:
        if self._data is not None:
            return self._data
        with self._lock:
            if self._data is not None:
                return self._data
            if not self._path.exists():
                log.warning("Disease knowledge base missing; serving derived names only",
                            fields={"path": str(self._path)})
                self._data = {"diseases": {}, "disclaimer": "", "sources": []}
            else:
                self._data = json.loads(self._path.read_text(encoding="utf-8"))
                log.info("Loaded disease knowledge base",
                         fields={"classes": len(self._data.get("diseases", {}))})
            return self._data

    def reload(self) -> None:
        """Drop the cache so the next access re-reads from disk."""
        with self._lock:
            self._data = None

    # -------------------------------------------------------------- lookup
    def get(self, class_name: str) -> dict:
        """Full entry for a class, or a derived placeholder."""
        entry = self._ensure().get("diseases", {}).get(class_name)
        if entry:
            return entry
        plant, condition, healthy = _pretty(class_name)
        return {
            "class_name": class_name,
            "common_name": condition,
            "display_name": f"{plant} - {condition}",
            "plant": plant,
            "condition": condition,
            "is_healthy": healthy,
            "category": "healthy" if healthy else "unknown",
            "severity": "none" if healthy else "unknown",
            "description": (
                "No curated information is available for this class yet. "
                "The name is derived from the dataset label."
            ),
            "symptoms": [],
            "causes": [],
            "favourable_conditions": [],
            "prevention": [],
            "management": [],
            "disclaimer": self.disclaimer,
            "sources": self.sources,
        }

    def display_name(self, class_name: str) -> str:
        return self.get(class_name)["display_name"]

    def plant_of(self, class_name: str) -> str:
        return self.get(class_name)["plant"]

    def condition_of(self, class_name: str) -> str:
        return self.get(class_name)["condition"]

    def is_healthy(self, class_name: str) -> bool:
        return bool(self.get(class_name)["is_healthy"])

    def severity(self, class_name: str) -> str:
        return self.get(class_name).get("severity", "unknown")

    # ---------------------------------------------------------------- list
    def all(self) -> list[dict]:
        """Every curated entry, sorted by plant then condition."""
        entries = list(self._ensure().get("diseases", {}).values())
        return sorted(entries, key=lambda e: (e.get("plant", ""), e.get("condition", "")))

    def summaries(self) -> list[dict]:
        """Compact rows for the disease-library list view."""
        return [
            {
                "class_name": e["class_name"],
                "display_name": e["display_name"],
                "common_name": e["common_name"],
                "plant": e["plant"],
                "condition": e["condition"],
                "category": e.get("category", "unknown"),
                "severity": e.get("severity", "unknown"),
                "is_healthy": e.get("is_healthy", False),
                "description": e.get("description", ""),
                "symptom_count": len(e.get("symptoms", [])),
                "training_images": e.get("training_images"),
            }
            for e in self.all()
        ]

    def plants(self) -> list[str]:
        return sorted({e["plant"] for e in self.all()})

    def categories(self) -> list[str]:
        return sorted({e.get("category", "unknown") for e in self.all()})

    # -------------------------------------------------------------- extras
    @property
    def disclaimer(self) -> str:
        return self._ensure().get("disclaimer", "")

    @property
    def sources(self) -> list[str]:
        return self._ensure().get("sources", [])

    @property
    def metadata(self) -> dict:
        data = self._ensure()
        return {
            "version": data.get("version"),
            "generated_at": data.get("generated_at"),
            "class_count": data.get("class_count", len(data.get("diseases", {}))),
            "disclaimer": data.get("disclaimer", ""),
            "sources": data.get("sources", []),
            "editorial_policy": data.get("editorial_policy", ""),
            "severity_scale": data.get("severity_scale", {}),
        }


knowledge_base = KnowledgeBase()
