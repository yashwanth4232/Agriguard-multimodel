"""M2-003 verification scaffold; create during M2-002 but do not execute yet.

When authorized, this script must exercise live Open-Meteo success separately
from deterministic mocked response, failure, validation, and artifact checks.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from argparse import ArgumentParser
from pathlib import Path
from urllib.error import URLError


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("KERAS_HOME", str(PROJECT_ROOT / ".keras"))

import config  # noqa: E402
import context  # noqa: E402


PROTECTED_ARTIFACTS = (
    config.MODEL_PATH,
    config.CLASS_NAMES_PATH,
    config.TRAINING_HISTORY_PATH,
    config.EXPERIMENT_CONFIG_PATH,
    config.TEST_METRICS_PATH,
    config.CLASSIFICATION_REPORT_PATH,
    config.CONFUSION_MATRIX_CSV_PATH,
    config.TRAINING_ACCURACY_FIG_PATH,
    config.TRAINING_LOSS_FIG_PATH,
    config.CONFUSION_MATRIX_FIG_PATH,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protected_hashes() -> dict[str, str]:
    return {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in PROTECTED_ARTIFACTS}


def valid_prediction() -> dict[str, object]:
    labels = context.load_authoritative_class_names()
    index = labels.index("Tomato___Early_blight")
    return {
        "predicted_index": index,
        "predicted_label": labels[index],
        "confidence": 0.9400433,
        "output_dimension": len(labels),
    }


class FakeResponse:
    """Minimal context-manager response for deterministic weather parsing tests."""

    status = 200

    def __init__(self, payload: object) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def fake_urlopen(payload: object):
    def opener(_url: str, *, timeout: float) -> FakeResponse:
        assert timeout > 0
        return FakeResponse(payload)

    return opener


def expect_error(callback, message: str) -> bool:
    try:
        callback()
    except (context.ContextValidationError, context.WeatherRequestError) as exc:
        return message in str(exc)
    return False


def run_deterministic_checks() -> dict[str, bool]:
    """M2-003 checks that do not need network access or Model 1 execution."""
    payload = {
        "current": {
            "time": "2026-08-13T12:00",
            "temperature_2m": 29.4,
            "relative_humidity_2m": 81.0,
            "precipitation": 2.3,
            "weather_code": 61,
        },
        "current_units": {
            "temperature_2m": "°C",
            "relative_humidity_2m": "%",
            "precipitation": "mm",
            "weather_code": "wmo code",
        },
    }
    weather = context.get_weather(
        17.3850, 78.4867, urlopen_func=fake_urlopen(payload)
    )
    prediction = valid_prediction()
    crop, disease = context.extract_crop_and_disease(prediction["predicted_label"])
    location = context.validate_location(
        {"source": "user_selected", "latitude": 17.3850, "longitude": 78.4867}
    )
    unavailable = context.validate_location(None)
    api_failure = expect_error(
        lambda: context.get_weather(
            17.3850,
            78.4867,
            urlopen_func=lambda _url, *, timeout: (_ for _ in ()).throw(URLError("offline")),
        ),
        "Weather request failed",
    )
    missing_field = dict(payload)
    missing_field["current"] = dict(payload["current"])
    del missing_field["current"]["precipitation"]
    return {
        "valid_prediction": context.validate_model1_prediction(prediction)["predicted_label"]
        == prediction["predicted_label"],
        "valid_class_mapping": len(context.load_authoritative_class_names()) == 38,
        "crop_extraction": crop == "Tomato" and disease == "Tomato___Early_blight",
        "confidence_preserved": prediction["confidence"] == 0.9400433,
        "growth_stages_accepted": all(
            context.validate_growth_stage(stage) == stage
            for stage in context.GROWTH_STAGE_DESCRIPTIONS
        ),
        "invalid_growth_stage_rejected": expect_error(
            lambda: context.validate_growth_stage("Bud"), "Invalid growth stage"
        ),
        "valid_location": location["location_source"] == "user_selected",
        "invalid_latitude_rejected": expect_error(
            lambda: context.validate_location(
                {"source": "user_selected", "latitude": 91, "longitude": 0}
            ),
            "Invalid latitude",
        ),
        "invalid_longitude_rejected": expect_error(
            lambda: context.validate_location(
                {"source": "user_selected", "latitude": 0, "longitude": 181}
            ),
            "Invalid longitude",
        ),
        "unavailable_location": unavailable["location_status"] == "unavailable",
        "device_gps_confirmation_required": expect_error(
            lambda: context.validate_location(
                {"source": "device_gps", "latitude": 0, "longitude": 0}
            ),
            "at_crop_location_confirmed",
        ),
        "weather_response_parsed": weather["weather_status"] == "available",
        "weather_units_normalized": (
            weather["temperature_c"] == 29.4
            and weather["humidity_percent"] == 81.0
            and weather["precipitation_mm"] == 2.3
        ),
        "weather_condition_preserved": weather["weather_condition"] == 61,
        "weather_evidence_recorded": bool(weather["weather_query_parameters"])
        and bool(weather["weather_request_timestamp"]),
        "api_failure_handled": api_failure,
        "missing_weather_field_handled": expect_error(
            lambda: context.get_weather(
                17.3850, 78.4867, urlopen_func=fake_urlopen(missing_field)
            ),
            "missing current fields",
        ),
    }


def main() -> None:
    """Run only after M2-003 authorization; live weather is opt-in."""
    parser = ArgumentParser()
    parser.add_argument(
        "--live-weather",
        action="store_true",
        help="Also perform the required controlled-coordinate Open-Meteo request.",
    )
    arguments = parser.parse_args()
    before = protected_hashes()
    checks = run_deterministic_checks()
    if arguments.live_weather:
        live_weather = context.get_weather(17.3850, 78.4867)
        checks["live_open_meteo_request"] = live_weather["weather_status"] == "available"
    after = protected_hashes()
    checks["protected_artifacts_unchanged"] = before == after
    print(json.dumps({"checks": checks, "protected_hashes": after}, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
