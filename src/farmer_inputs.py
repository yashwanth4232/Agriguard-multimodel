"""Farmer-input data contract for AgriGuard agricultural context.

This module is intentionally UI- and hardware-independent. A future mobile or
web UI supplies one chosen growth stage plus coordinates from either confirmed
device GPS or a user-selected crop-location picker. It never reads a computer,
uploader, IP, or image location.
"""

from __future__ import annotations

from typing import Mapping

import context


FARMER_GROWTH_STAGE_DESCRIPTIONS = {
    "Seedling": "Newly emerged plant with small young leaves.",
    "Vegetative": "Plant is mainly developing leaves and stems.",
    "Flowering": "Flowers are appearing on the plant.",
    "Fruiting": "Fruits are forming or developing.",
    "Maturity": "Crop or fruit is approaching harvest.",
    "Unknown": "Farmer is not sure of the growth stage.",
}


def select_growth_stage(selected_stage: str) -> dict[str, str]:
    """Return one permitted farmer selection and its application-facing help text."""
    context.validate_growth_stage(selected_stage)
    return {
        "growth_stage": selected_stage,
        "growth_stage_description": FARMER_GROWTH_STAGE_DESCRIPTIONS[selected_stage],
    }


def device_gps_location(
    latitude: float,
    longitude: float,
    *,
    at_crop_location_confirmed: bool,
) -> dict[str, object]:
    """Accept future device-GPS coordinates only after explicit farmer confirmation."""
    return context.validate_location(
        {
            "source": "device_gps",
            "latitude": latitude,
            "longitude": longitude,
            "at_crop_location_confirmed": at_crop_location_confirmed,
        }
    )


def user_selected_location(latitude: float, longitude: float) -> dict[str, object]:
    """Accept coordinates supplied by a future crop-location map/picker workflow."""
    return context.validate_location(
        {"source": "user_selected", "latitude": latitude, "longitude": longitude}
    )


def unavailable_crop_location() -> dict[str, object]:
    """Represent an explicit no-location choice without any coordinate inference."""
    return context.validate_location(None)


def build_farmer_context_input(
    selected_stage: str,
    location: Mapping[str, object] | None,
) -> dict[str, object]:
    """Combine validated farmer inputs without requesting weather or Model 1 inference."""
    return {**select_growth_stage(selected_stage), **context.validate_location(location)}
