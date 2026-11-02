"""Farmer-facing Flask application layer for the read-only AgriGuard pipeline."""

from __future__ import annotations

import secrets
from pathlib import Path

from flask import Flask, abort, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
import sys

sys.path.insert(0, str(SRC_DIR))

import advisory  # noqa: E402
import context  # noqa: E402
import farmer_inputs  # noqa: E402
import fusion  # noqa: E402
import ood  # noqa: E402
import predict  # noqa: E402
import xai  # noqa: E402


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RUNTIME_ROOT = PROJECT_ROOT / "instance" / "app_runtime"
UPLOADS_DIR = RUNTIME_ROOT / "uploads"
GENERATED_DIR = PROJECT_ROOT / "static" / "generated"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


def _prepare_runtime_directories() -> None:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)


def _save_upload() -> tuple[str, Path]:
    uploaded = request.files.get("leaf_image")
    if uploaded is None or not uploaded.filename:
        raise ValueError("Please choose a crop-leaf image.")
    filename = secure_filename(uploaded.filename)
    if not filename or Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError("Please upload a JPG, JPEG, PNG, BMP, or WEBP image.")
    run_id = secrets.token_hex(12)
    destination = UPLOADS_DIR / f"{run_id}_{filename}"
    uploaded.save(destination)
    return run_id, destination


def _location_from_form() -> dict[str, object]:
    """Translate the farmer-facing location mode into the internal contract."""
    mode = (request.form.get("location_mode") or "none").strip()
    latitude = (request.form.get("latitude") or "").strip()
    longitude = (request.form.get("longitude") or "").strip()
    confirmation = request.form.get("at_crop_location_confirmed") == "true"
    if mode == "none":
        return farmer_inputs.unavailable_crop_location()
    if mode == "manual":
        if not latitude or not longitude:
            raise ValueError("Enter both latitude and longitude for Manual Location.")
        try:
            return farmer_inputs.user_selected_location(float(latitude), float(longitude))
        except (TypeError, ValueError) as exc:
            raise ValueError("Enter valid geographic coordinates for Manual Location.") from exc
    if mode == "auto":
        if not latitude or not longitude:
            raise ValueError("Use My Current Location before submitting Auto Location.")
        if not confirmation:
            raise ValueError("Confirm that your current device location is the crop or field location.")
        try:
            return farmer_inputs.device_gps_location(
                float(latitude), float(longitude), at_crop_location_confirmed=True
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("Your device location is not a valid geographic location.") from exc
    raise ValueError("Choose Auto Location, Manual Location, or No Location.")


def _location_context_input(location: dict[str, object]) -> dict[str, object] | None:
    """Convert an already validated application location to Context's input form."""
    if location["location_status"] == "unavailable":
        return None
    context_input: dict[str, object] = {
        "source": location["location_source"],
        "latitude": location["latitude"],
        "longitude": location["longitude"],
    }
    if location["location_source"] == "device_gps":
        context_input["at_crop_location_confirmed"] = True
    return context_input


def _friendly_error(exc: Exception) -> str:
    """Never expose internal validation contracts on the farmer-facing page."""
    message = str(exc)
    friendly_messages = {
        "Enter both latitude and longitude for Manual Location.",
        "Enter valid geographic coordinates for Manual Location.",
        "Use My Current Location before submitting Auto Location.",
        "Confirm that your current device location is the crop or field location.",
        "Your device location is not a valid geographic location.",
        "Choose Auto Location, Manual Location, or No Location.",
    }
    if message in friendly_messages:
        return message
    if "latitude" in message.lower() or "longitude" in message.lower() or "coordinate" in message.lower():
        return "Enter valid geographic coordinates for Manual Location."
    if "location" in message.lower():
        return "Please check your selected location option and try again."
    return message


@app.get("/")
def index():
    return render_template(
        "index.html",
        stages=farmer_inputs.FARMER_GROWTH_STAGE_DESCRIPTIONS,
    )


@app.post("/assess")
def assess():
    _prepare_runtime_directories()
    try:
        stage = farmer_inputs.select_growth_stage(request.form.get("growth_stage", ""))
        location = _location_from_form()
        location_context_input = _location_context_input(location)
        run_id, image_path = _save_upload()
        model, class_names = predict.load_prediction_artifacts()
        domain = ood.domain_result(image_path, model)
        if not domain["accepted"]:
            return render_template(
                "index.html", stages=farmer_inputs.FARMER_GROWTH_STAGE_DESCRIPTIONS,
                error="This image is outside AgriGuard's supported crop-leaf domain. Please upload a clear crop-leaf photograph."
            ), 400
        prediction = predict.predict_image(image_path, model, class_names)
        xai_metadata = xai.generate_gradcam_evidence(
            image_path, output_directory=GENERATED_DIR / run_id
        )
        normalized_context = context.build_context(
            prediction, stage["growth_stage"], location_context_input
        )
        fusion_record = fusion.fuse_evidence(
            prediction,
            normalized_context,
            xai_metadata,
            model_artifact_sha256=xai_metadata["model_sha256"],
        )
        advisory_result = advisory.build_advisory(fusion_record)
    except (ValueError, context.ContextValidationError, fusion.FusionValidationError, advisory.AdvisoryValidationError) as exc:
        return render_template(
            "index.html", stages=farmer_inputs.FARMER_GROWTH_STAGE_DESCRIPTIONS, error=_friendly_error(exc)
        ), 400

    image_url = f"/uploads/{image_path.name}"
    overlay_reference = str(xai_metadata["artifacts"]["overlay"])
    overlay_url = "/static/" + Path(overlay_reference).relative_to(PROJECT_ROOT / "static").as_posix()
    return render_template(
        "result.html",
        prediction=prediction,
        crop=normalized_context["crop"],
        disease=normalized_context["disease"],
        context=normalized_context,
        stage=stage,
        advisory=advisory_result,
        image_url=image_url,
        overlay_url=overlay_url,
        limitations=[
            "Model 1 prediction is not independently confirmed by Model 2.",
            "Current weather is current context, not necessarily weather at image-capture time.",
            "Grad-CAM is an attribution visualization, not proof of biological causation.",
            "Advisory guidance is source-grounded and does not guarantee treatment success.",
        ],
    )


@app.get("/uploads/<path:filename>")
def uploaded_image(filename: str):
    if Path(filename).name != filename:
        abort(404)
    return send_from_directory(UPLOADS_DIR, filename)


@app.errorhandler(413)
def too_large(_error):
    return render_template(
        "index.html", stages=farmer_inputs.FARMER_GROWTH_STAGE_DESCRIPTIONS,
        error="Image is too large. Maximum upload size is 10 MB."
    ), 413


if __name__ == "__main__":
    _prepare_runtime_directories()
    app.run(host="127.0.0.1", port=5000, debug=False)
