"""Deterministic, source-grounded advisory evidence selection for AgriGuard."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping


ADVISORY_TYPE = "deterministic_source_grounded"
POLICY_VERSION = "1.0.0"
SAFETY_DISCLAIMER = (
    "This source-grounded information is associated with the Model 1 prediction and "
    "available current context. It does not confirm the diagnosis, establish cause, "
    "or guarantee crop recovery. Follow locally applicable agricultural and product-label guidance."
)
LIMITATIONS = [
    "No confidence threshold policy is defined; the exact Model 1 confidence is preserved without a confidence band.",
    "Current weather is context only and is not weather at image capture time.",
    "Knowledge-base evidence remains pending human review.",
]


class AdvisoryValidationError(ValueError):
    """Raised when fusion, taxonomy, knowledge-base, or source evidence is inconsistent."""


def _load_json(path: str | Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AdvisoryValidationError(f"Invalid JSON object: {path}")
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_knowledge_assets(
    knowledge_base_path: str | Path = "data/advisory/knowledge_base.json",
    source_registry_path: str | Path = "data/advisory/source_registry.json",
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    """Load versioned, curated evidence and source provenance without mutation."""
    knowledge_base = _load_json(knowledge_base_path)
    registry = _load_json(source_registry_path)
    sources = registry.get("sources")
    if not isinstance(sources, list):
        raise AdvisoryValidationError("Invalid source registry: sources must be a list.")
    return knowledge_base, {str(source["source_id"]): source for source in sources if isinstance(source, dict) and "source_id" in source}


def _validated_fusion(fusion_record: Mapping[str, object]) -> tuple[Mapping[str, object], Mapping[str, object], Mapping[str, object]]:
    if fusion_record.get("fusion_type") != "deterministic_context_association":
        raise AdvisoryValidationError("Invalid fusion record: unsupported fusion_type.")
    diagnosis = fusion_record.get("diagnosis")
    context = fusion_record.get("agricultural_context")
    visual = fusion_record.get("visual_explanation")
    if not all(isinstance(item, Mapping) for item in (diagnosis, context, visual)):
        raise AdvisoryValidationError("Invalid fusion record: diagnosis, context, and visual_explanation are required mappings.")
    assert isinstance(diagnosis, Mapping) and isinstance(context, Mapping) and isinstance(visual, Mapping)
    if diagnosis.get("source") != "Model 1" or diagnosis.get("unchanged_by_fusion") is not True:
        raise AdvisoryValidationError("Invalid fusion diagnosis provenance.")
    label = diagnosis.get("predicted_label")
    if not isinstance(label, str) or diagnosis.get("disease") != label:
        raise AdvisoryValidationError("Invalid fusion diagnosis label/disease consistency.")
    if context.get("crop") != diagnosis.get("crop") or context.get("disease") != label:
        raise AdvisoryValidationError("Invalid fusion context consistency with Model 1 diagnosis.")
    if context.get("model1_confidence") != diagnosis.get("confidence"):
        raise AdvisoryValidationError("Invalid fusion context Model 1 confidence consistency.")
    if visual.get("available") is True and (visual.get("target_class") != label or visual.get("target_class_index") != diagnosis.get("predicted_index")):
        raise AdvisoryValidationError("Invalid fusion XAI consistency with Model 1 diagnosis.")
    return diagnosis, context, visual


def build_advisory(
    fusion_record: Mapping[str, object],
    *,
    knowledge_base_path: str | Path = "data/advisory/knowledge_base.json",
    source_registry_path: str | Path = "data/advisory/source_registry.json",
    generated_at: str | None = None,
) -> dict[str, object]:
    """Select only exact, curated evidence; never diagnose or generate recommendations."""
    diagnosis, current_context, visual = _validated_fusion(fusion_record)
    knowledge_base, registry = load_knowledge_assets(knowledge_base_path, source_registry_path)
    entries = knowledge_base.get("entries")
    if not isinstance(entries, list):
        raise AdvisoryValidationError("Invalid knowledge base: entries must be a list.")
    matching = [item for item in entries if isinstance(item, dict) and item.get("model1_label") == diagnosis["predicted_label"]]
    if len(matching) != 1:
        raise AdvisoryValidationError("Taxonomy mismatch: exact Model 1 label must occur exactly once in knowledge base.")
    entry = matching[0]
    if entry.get("crop") != diagnosis.get("crop"):
        raise AdvisoryValidationError("Taxonomy mismatch: knowledge-base crop does not match fusion diagnosis.")
    if entry.get("condition_type") == "healthy" and entry.get("source_status") == "supported":
        pass
    growth_stage = current_context.get("growth_stage")
    selected, excluded, source_ids = [], [], set()
    supported_status = entry.get("source_status")
    if supported_status != "unsupported":
        for guidance in entry.get("guidance_entries", []):
            stages = guidance.get("growth_stages", [])
            if guidance.get("review_status") != "pending_human_review":
                excluded.append({"guidance_id": guidance.get("guidance_id"), "reason": "review_status_not_permitted"})
            elif stages and (growth_stage == "Unknown" or growth_stage not in stages):
                excluded.append({"guidance_id": guidance.get("guidance_id"), "reason": "growth_stage_not_explicitly_supported"})
            elif guidance.get("source_id") not in registry or not guidance.get("source_reference"):
                raise AdvisoryValidationError("Guidance source provenance is missing or unknown.")
            else:
                selected.append(dict(guidance)); source_ids.add(str(guidance["source_id"]))
    status = "unavailable" if supported_status == "unsupported" or not selected else ("available" if supported_status == "supported" else "partially_available")
    if entry.get("condition_type") == "healthy":
        if any(item.get("category") == "management" for item in selected):
            raise AdvisoryValidationError("Healthy classes cannot receive disease-management guidance.")
    return {
        "advisory_type": ADVISORY_TYPE,
        "diagnosis": {"source": "Model 1", "predicted_label": diagnosis["predicted_label"], "predicted_index": diagnosis["predicted_index"], "confidence": diagnosis["confidence"], "unchanged": True},
        "current_context": dict(current_context),
        "visual_explanation": dict(visual),
        "advisory_guidance": {"status": status, "condition_type": entry.get("condition_type"), "entries": selected, "selected_guidance_ids": [item["guidance_id"] for item in selected], "excluded_guidance": excluded},
        "sources": [registry[source_id] for source_id in sorted(source_ids)],
        "availability": {"weather_available": current_context.get("weather_status") == "available", "location_available": current_context.get("location_status") == "available", "growth_stage_known": growth_stage != "Unknown", "xai_available": visual.get("available") is True, "advisory_available": status != "unavailable"},
        "limitations": [*fusion_record.get("limitations", []), *LIMITATIONS],
        "provenance": {"fusion_type": fusion_record["fusion_type"], "knowledge_base_version": knowledge_base.get("knowledge_base_version"), "knowledge_base_review_status": knowledge_base.get("review_status"), "source_registry_schema_version": _load_json(source_registry_path).get("schema_version"), "policy_version": POLICY_VERSION, "generated_at": generated_at or _utc_now()},
        "safety_disclaimer": SAFETY_DISCLAIMER,
    }
