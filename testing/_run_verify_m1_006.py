import json
import runpy
import sys
from pathlib import Path

out = Path(__file__).resolve().parent.parent / "tmp" / "m1_006_result.json"
try:
    runpy.run_path(str(Path(__file__).resolve().parent / "verify_m1_006_model.py"), run_name="__main__")
    out.write_text(json.dumps({"status": "passed"}), encoding="utf-8")
except SystemExit as exc:
    out.write_text(json.dumps({"status": "failed", "exit_code": exc.code}), encoding="utf-8")
    raise
