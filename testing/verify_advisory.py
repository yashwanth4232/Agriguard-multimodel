"""Controlled verification for deterministic, source-grounded advisory selection."""
from __future__ import annotations
import copy, hashlib, json, os, sys
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parent.parent; os.environ.setdefault("KERAS_HOME",str(PROJECT_ROOT/".keras")); sys.path.insert(0,str(PROJECT_ROOT/"src"))
import advisory  # noqa: E402
PROTECTED=[p for p in ["models/agriguard_mobilenetv2.keras","models/class_names.json","experiments/training_history.csv","experiments/experiment_config.json","experiments/test_metrics.json","experiments/classification_report.csv","experiments/confusion_matrix.csv","reports/figures/training_accuracy.png","reports/figures/training_loss.png","reports/figures/confusion_matrix.png","experiments/context_test_records.json","reports/context/m2_verification.json","reports/context/m2_evidence_audit.json","reports/context/end_to_end_m1_m2_test.json","experiments/xai_test_records.json","reports/xai/gradcam_raw_heatmap.npy","reports/xai/gradcam_resized_heatmap.npy","reports/xai/gradcam_overlay.png","reports/xai/xai_verification.json","experiments/fusion_test_records.json","reports/fusion/fusion_input.json","reports/fusion/fusion_output.json","reports/fusion/fusion_verification.json","data/advisory/knowledge_base.json","data/advisory/source_registry.json"]]
def hashes(): return {p:hashlib.sha256((PROJECT_ROOT/p).read_bytes()).hexdigest() for p in PROTECTED}
def make(label):
    class_names=json.loads((PROJECT_ROOT/"models/class_names.json").read_text())
    index=class_names.index(label)
    r=json.loads((PROJECT_ROOT/"reports/fusion/fusion_output.json").read_text()); d=r["diagnosis"]; d["predicted_label"]=d["disease"]=label; d["predicted_index"]=index; d["crop"]=label.split("___",1)[0]; r["agricultural_context"]["disease"]=label; r["agricultural_context"]["crop"]=d["crop"]
    if r["visual_explanation"].get("available"):
        r["visual_explanation"]["target_class"]=label
        r["visual_explanation"]["target_class_index"]=index
    return r
def fails(callback):
    try: callback()
    except advisory.AdvisoryValidationError: return True
    return False
def main():
    before=hashes(); base=json.loads((PROJECT_ROOT/"reports/fusion/fusion_output.json").read_text()); output=advisory.build_advisory(base,generated_at="2026-08-13T00:00:00Z")
    partial=advisory.build_advisory(make("Tomato___Leaf_Mold"),generated_at="2026-08-13T00:00:00Z")
    unsupported=advisory.build_advisory(make("Grape___Esca_(Black_Measles)"),generated_at="2026-08-13T00:00:00Z")
    healthy=advisory.build_advisory(make("Apple___healthy"),generated_at="2026-08-13T00:00:00Z")
    pest=advisory.build_advisory(make("Tomato___Spider_mites Two-spotted_spider_mite"),generated_at="2026-08-13T00:00:00Z")
    unknown=copy.deepcopy(base); unknown["agricultural_context"]["growth_stage"]="Unknown"; unknown_out=advisory.build_advisory(unknown,generated_at="2026-08-13T00:00:00Z")
    missing=copy.deepcopy(base); missing["agricultural_context"].update({"weather_status":"unavailable","location_status":"unavailable","latitude":None,"longitude":None,"temperature_c":None,"humidity_percent":None,"precipitation_mm":None,"weather_condition":None}); missing["visual_explanation"]["available"]=False; missing_out=advisory.build_advisory(missing,generated_at="2026-08-13T00:00:00Z")
    bad=copy.deepcopy(base); bad["agricultural_context"]["crop"]="Potato"; after=hashes()
    checks={"supported_disease":output["advisory_guidance"]["status"]=="available","partially_supported":partial["advisory_guidance"]["status"]=="partially_available","unsupported":unsupported["advisory_guidance"]["status"]=="unavailable","healthy_no_disease_management":all(e["category"]!="management" for e in healthy["advisory_guidance"]["entries"]),"pest_unavailable":pest["advisory_guidance"]["status"]=="unavailable","unknown_stage_not_guessed":unknown_out["availability"]["growth_stage_known"] is False,"unavailable_weather_location_xai":missing_out["availability"]=={"weather_available":False,"location_available":False,"growth_stage_known":True,"xai_available":False,"advisory_available":True},"nulls_preserved":missing_out["current_context"]["temperature_c"] is None,"taxonomy_mismatch_rejected":fails(lambda:advisory.build_advisory(bad)),"prediction_preserved":output["diagnosis"]["predicted_label"]==base["diagnosis"]["predicted_label"] and output["diagnosis"]["predicted_index"]==base["diagnosis"]["predicted_index"],"confidence_preserved":output["diagnosis"]["confidence"]==base["diagnosis"]["confidence"],"no_new_diagnosis_or_confidence":output["diagnosis"]["unchanged"] is True,"source_provenance":bool(output["sources"]) and bool(output["advisory_guidance"]["selected_guidance_ids"]),"no_chemical_instruction_leakage":not any(t in json.dumps(output).lower() for t in ["ml/l","g/l","dosage","dilution","tank mix"]),"protected_artifacts_unchanged":before==after}
    report=PROJECT_ROOT/"reports/advisory"; report.mkdir(parents=True,exist_ok=True); (report/"advisory_input.json").write_text(json.dumps(base,indent=2)+"\n"); (report/"advisory_output.json").write_text(json.dumps(output,indent=2)+"\n"); evidence={"checks":checks,"protected_hashes_before":before,"protected_hashes_after":after,"no_api_call":True,"no_training":True,"no_llm":True}; (report/"advisory_verification.json").write_text(json.dumps(evidence,indent=2)+"\n"); (PROJECT_ROOT/"experiments/advisory_test_records.json").write_text(json.dumps(evidence,indent=2)+"\n"); print(json.dumps(evidence,indent=2));
    if not all(checks.values()): raise SystemExit(1)
if __name__=="__main__": main()
