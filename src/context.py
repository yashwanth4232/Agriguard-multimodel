"""Agricultural-context acquisition for AgriGuard Modality 2.

This module consumes the dictionary returned by :func:`predict.predict_image`.
It does not load, train, alter, or otherwise depend on the Model 1 model at
runtime.  Weather is current context for the supplied crop/field coordinates,
not weather at the time an image was captured.
"""

from __future__ import annotations

import json
import math
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import config


OPEN_METEO_PROVIDER = "Open-Meteo"
OPEN_METEO_DOCUMENTATION_URL = "https://open-meteo.com/en/docs"
OPEN_METEO_ENDPOINT = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_CURRENT_VARIABLES = (
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "weather_code",
)
OPEN_METEO_UNITS = {
    "temperature_2m": "°C",
    "relative_humidity_2m": "%",
    "precipitation": "mm",
    "weather_code": "wmo code",
}
PRECIPITATION_SEMANTICS = (
    "Open-Meteo current-condition precipitation (rain + showers + snow) in mm; "
    "it is not daily rainfall. Open-Meteo documents current conditions as based "
    "on 15-minutely weather-model data."
)
GROWTH_STAGE_DESCRIPTIONS = {
    "Seedling": "Small, young plant with a few leaves.",
    "Vegetative": "Plant is mainly growing leaves and stems.",
    "Flowering": "Flowers are appearing on the plant.",
    "Fruiting": "Fruits are forming or growing.",
    "Maturity": "Crop is fully developed and nearing harvest.",
    "Unknown": "The farmer does not know the current growth stage.",
}
LOCATION_SOURCES = {"user_selected", "device_gps"}


class ContextValidationError(ValueError):
    """Raised when required Model 1, growth-stage, or location input is invalid."""


class WeatherRequestError(RuntimeError):
    """Raised when current weather cannot be retrieved or safely normalized."""


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContextValidationError(f"Invalid {field_name}: expected a finite numeric value.")
    number = float(value)
    if not math.isfinite(number):
        raise ContextValidationError(f"Invalid {field_name}: expected a finite numeric value.")
    return number


def load_authoritative_class_names() -> list[str]:
    """Load the exact saved Model 1 class order, rejecting malformed mappings."""
    path = config.CLASS_NAMES_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Saved class names not found: {path}")
    try:
        class_names = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ContextValidationError(f"Invalid authoritative class mapping: {path}") from exc
    if (
        not isinstance(class_names, list)
        or len(class_names) != 38
        or any(not isinstance(label, str) or not label for label in class_names)
    ):
        raise ContextValidationError(
            f"Expected 38 non-empty class names in authoritative mapping: {path}."
        )
    return class_names


def validate_model1_prediction(model1_prediction: Mapping[str, object]) -> dict[str, object]:
    """Validate the public ``predict.predict_image`` result against Model 1 evidence."""
    if not isinstance(model1_prediction, Mapping):
        raise ContextValidationError("Invalid Model 1 prediction: expected a mapping.")

    required = {"predicted_index", "predicted_label", "confidence", "output_dimension"}
    missing = sorted(required - set(model1_prediction))
    if missing:
        raise ContextValidationError(
            f"Invalid Model 1 prediction: missing required fields: {', '.join(missing)}."
        )

    class_names = load_authoritative_class_names()
    label = model1_prediction["predicted_label"]
    index = model1_prediction["predicted_index"]
    output_dimension = model1_prediction["output_dimension"]
    if not isinstance(label, str) or label not in class_names:
        raise ContextValidationError(
            "Invalid Model 1 prediction: predicted_label is not in models/class_names.json."
        )
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(class_names):
        raise ContextValidationError("Invalid Model 1 prediction: predicted_index is out of range.")
    if class_names[index] != label:
        raise ContextValidationError(
            "Invalid Model 1 prediction: predicted_index does not match predicted_label."
        )
    if isinstance(output_dimension, bool) or not isinstance(output_dimension, int):
        raise ContextValidationError("Invalid Model 1 prediction: output_dimension must be an integer.")
    if output_dimension != len(class_names):
        raise ContextValidationError(
            "Invalid Model 1 prediction: output_dimension does not match authoritative class count."
        )

    confidence = _finite_number(model1_prediction["confidence"], "Model 1 confidence")
    if not 0.0 <= confidence <= 1.0:
        raise ContextValidationError("Invalid Model 1 confidence: value must be between 0 and 1.")
    return {
        "predicted_index": index,
        "predicted_label": label,
        "confidence": model1_prediction["confidence"],
        "output_dimension": output_dimension,
    }


def extract_crop_and_disease(predicted_label: str) -> tuple[str, str]:
    """Derive crop and preserve disease-category label using the validated delimiter."""
    if predicted_label.count("___") != 1:
        raise ContextValidationError(
            "Invalid Model 1 class label: expected exactly one '___' crop/disease separator."
        )
    crop, disease_component = predicted_label.split("___", 1)
    if not crop or not disease_component:
        raise ContextValidationError("Invalid Model 1 class label: crop and disease must be non-empty.")
    return crop, predicted_label


def validate_growth_stage(growth_stage: str) -> str:
    """Accept exactly the farmer-facing growth stages specified for M2."""
    if growth_stage not in GROWTH_STAGE_DESCRIPTIONS:
        expected = ", ".join(GROWTH_STAGE_DESCRIPTIONS)
        raise ContextValidationError(f"Invalid growth stage: expected {expected}.")
    return growth_stage


def unavailable_location() -> dict[str, object]:
    """Return the explicit no-location representation; never infer coordinates."""
    return {
        "location_status": "unavailable",
        "location_source": "unavailable",
        "latitude": None,
        "longitude": None,
    }


def validate_location(location: Mapping[str, object] | None) -> dict[str, object]:
    """Validate a crop/field location supplied by a user-selected picker or device GPS.

    A device-GPS location is accepted only when its caller sets
    ``at_crop_location_confirmed`` to ``True``. This prevents the device or
    uploader location from being silently treated as the crop location.
    A user-selected location represents coordinates returned by a future
    map/location picker. No image metadata or uploader/computer location is
    accepted as a crop location.
    """
    if location is None:
        return unavailable_location()
    if not isinstance(location, Mapping):
        raise ContextValidationError("Invalid location: expected a mapping or None.")
    if location.get("source") == "unavailable":
        if any(key in location and location[key] is not None for key in ("latitude", "longitude")):
            raise ContextValidationError("Invalid unavailable location: coordinates must be null.")
        return unavailable_location()

    source = location.get("source")
    if source not in LOCATION_SOURCES:
        allowed = ", ".join(sorted(LOCATION_SOURCES | {"unavailable"}))
        raise ContextValidationError(f"Invalid location source: expected {allowed}.")
    if source == "device_gps" and location.get("at_crop_location_confirmed") is not True:
        raise ContextValidationError(
            "Device GPS requires at_crop_location_confirmed=True; do not assume uploader location is crop location."
        )
    if "latitude" not in location or "longitude" not in location:
        raise ContextValidationError("Invalid location: latitude and longitude are required.")
    latitude = _finite_number(location["latitude"], "latitude")
    longitude = _finite_number(location["longitude"], "longitude")
    if not -90.0 <= latitude <= 90.0:
        raise ContextValidationError("Invalid latitude: value must be between -90 and 90.")
    if not -180.0 <= longitude <= 180.0:
        raise ContextValidationError("Invalid longitude: value must be between -180 and 180.")
    return {
        "location_status": "available",
        "location_source": source,
        "latitude": latitude,
        "longitude": longitude,
    }


def _validate_weather_response(payload: object) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise WeatherRequestError("Weather response malformed: expected a JSON object.")
    current = payload.get("current")
    current_units = payload.get("current_units")
    if not isinstance(current, Mapping) or not isinstance(current_units, Mapping):
        raise WeatherRequestError("Weather response malformed: missing current conditions or units.")
    required = (*OPEN_METEO_CURRENT_VARIABLES, "time")
    missing = [field for field in required if field not in current]
    if missing:
        raise WeatherRequestError(f"Weather response missing current fields: {', '.join(missing)}.")
    for field, expected_unit in OPEN_METEO_UNITS.items():
        if current_units.get(field) != expected_unit:
            raise WeatherRequestError(
                f"Weather response has incompatible unit for {field}: {current_units.get(field)!r}."
            )

    try:
        temperature = _finite_number(current["temperature_2m"], "weather temperature")
        humidity = _finite_number(current["relative_humidity_2m"], "weather humidity")
        precipitation = _finite_number(current["precipitation"], "weather precipitation")
        weather_code = _finite_number(current["weather_code"], "weather code")
    except ContextValidationError as exc:
        raise WeatherRequestError(str(exc)) from exc
    if not 0.0 <= humidity <= 100.0:
        raise WeatherRequestError("Invalid weather humidity: value must be between 0 and 100.")
    if precipitation < 0.0:
        raise WeatherRequestError("Invalid weather precipitation: value must not be negative.")
    if int(weather_code) != weather_code:
        raise WeatherRequestError("Invalid weather code: expected an integer WMO weather code.")
    timestamp = current["time"]
    if not isinstance(timestamp, str) or not timestamp:
        raise WeatherRequestError("Weather response has an invalid current timestamp.")
    return {
        "temperature_c": temperature,
        "humidity_percent": humidity,
        "precipitation_mm": precipitation,
        "weather_condition": int(weather_code),
        "context_timestamp": timestamp,
    }


def get_weather(
    latitude: float,
    longitude: float,
    *,
    timeout_seconds: float = 10.0,
    urlopen_func: Callable[..., Any] | None = None,
) -> dict[str, object]:
    """Retrieve and normalize current Open-Meteo weather for crop coordinates.

    The request uses documented ``current`` variables and requests Celsius,
    millimetres, and ISO-8601 timestamps explicitly. ``urlopen_func`` is an
    optional test seam; production callers should omit it.
    """
    validated = validate_location(
        {"source": "user_selected", "latitude": latitude, "longitude": longitude}
    )
    timeout = _finite_number(timeout_seconds, "weather timeout")
    if timeout <= 0:
        raise ContextValidationError("Invalid weather timeout: value must be greater than zero.")
    query_parameters = {
        "latitude": validated["latitude"],
        "longitude": validated["longitude"],
        "current": ",".join(OPEN_METEO_CURRENT_VARIABLES),
        "temperature_unit": "celsius",
        "precipitation_unit": "mm",
        "timeformat": "iso8601",
        "timezone": "GMT",
    }
    request_timestamp = _utc_timestamp()
    request_url = f"{OPEN_METEO_ENDPOINT}?{urlencode(query_parameters)}"
    opener = urlopen if urlopen_func is None else urlopen_func
    try:
        with opener(request_url, timeout=timeout) as response:
            status = getattr(response, "status", 200)
            if status != 200:
                raise WeatherRequestError(f"Weather request failed: HTTP status {status}.")
            raw_body = response.read().decode("utf-8")
    except HTTPError as exc:
        raise WeatherRequestError(f"Weather request failed: HTTP {exc.code}.") from exc
    except (URLError, socket.timeout, TimeoutError) as exc:
        reason = getattr(exc, "reason", exc)
        raise WeatherRequestError(f"Weather request failed: {reason}.") from exc
    except OSError as exc:
        raise WeatherRequestError(f"Weather request failed: {exc}.") from exc
    try:
        raw_response = json.loads(raw_body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WeatherRequestError("Weather response malformed: invalid JSON.") from exc

    normalized = _validate_weather_response(raw_response)
    return {
        **normalized,
        "weather_status": "available",
        "weather_provider": OPEN_METEO_PROVIDER,
        "weather_documentation_url": OPEN_METEO_DOCUMENTATION_URL,
        "weather_endpoint": OPEN_METEO_ENDPOINT,
        "weather_query_parameters": query_parameters,
        "weather_requested_variables": list(OPEN_METEO_CURRENT_VARIABLES),
        "weather_units": OPEN_METEO_UNITS.copy(),
        "precipitation_semantics": PRECIPITATION_SEMANTICS,
        "weather_request_timestamp": request_timestamp,
        "weather_raw_response": raw_response,
    }


def _unavailable_weather(error: str) -> dict[str, object]:
    return {
        "weather_status": "unavailable",
        "weather_provider": OPEN_METEO_PROVIDER,
        "weather_documentation_url": OPEN_METEO_DOCUMENTATION_URL,
        "weather_endpoint": OPEN_METEO_ENDPOINT,
        "weather_query_parameters": None,
        "weather_requested_variables": list(OPEN_METEO_CURRENT_VARIABLES),
        "weather_units": OPEN_METEO_UNITS.copy(),
        "precipitation_semantics": PRECIPITATION_SEMANTICS,
        "temperature_c": None,
        "humidity_percent": None,
        "precipitation_mm": None,
        "weather_condition": None,
        "context_timestamp": None,
        "weather_request_timestamp": None,
        "weather_raw_response": None,
        "weather_error": error,
    }


def build_context(
    model1_prediction: Mapping[str, object],
    growth_stage: str,
    location: Mapping[str, object] | None,
) -> dict[str, object]:
    """Build normalized current agricultural context without affecting Model 1.

    Weather failures and unavailable crop locations are represented explicitly
    in the returned context so that a valid visual prediction remains usable.
    Invalid Model 1, growth-stage, and location inputs are rejected clearly.
    """
    prediction = validate_model1_prediction(model1_prediction)
    stage = validate_growth_stage(growth_stage)
    normalized_location = validate_location(location)
    crop, disease = extract_crop_and_disease(str(prediction["predicted_label"]))
    if normalized_location["location_status"] == "unavailable":
        weather = _unavailable_weather("Location unavailable: no crop location was supplied.")
    else:
        try:
            weather = get_weather(
                float(normalized_location["latitude"]),
                float(normalized_location["longitude"]),
            )
        except WeatherRequestError as exc:
            weather = _unavailable_weather(str(exc))
    return {
        "crop": crop,
        "disease": disease,
        "model1_confidence": prediction["confidence"],
        "growth_stage": stage,
        "growth_stage_description": GROWTH_STAGE_DESCRIPTIONS[stage],
        **normalized_location,
        **weather,
        "image_capture_timestamp": model1_prediction.get("image_capture_timestamp"),
    }


def write_context_evidence(
    context: Mapping[str, object],
    *,
    raw_response_path: str | Path = "reports/context/api_sample_response.json",
    normalized_context_path: str | Path = "reports/context/normalized_context.json",
) -> None:
    """Persist raw weather and normalized context evidence when a caller requests it.

    The function creates only M2-specific directories. It does not run
    automatically, avoiding the creation of fabricated research evidence.
    """
    if not isinstance(context, Mapping):
        raise ContextValidationError("Invalid context evidence: expected a mapping.")
    raw_path = Path(raw_response_path)
    normalized_path = Path(normalized_context_path)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(
        json.dumps(context.get("weather_raw_response"), indent=2) + "\n",
        encoding="utf-8",
    )
    normalized_path.write_text(json.dumps(dict(context), indent=2) + "\n", encoding="utf-8")
