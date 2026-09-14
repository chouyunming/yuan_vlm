#!/usr/bin/env bash
# Deterministically enforce prompt_v2.py's R1..R11 rules over every
# frame_XXXXXX.json in a directory, in place.
#
# This is the SAME repair_response() that real inference (vlm_node.py)
# applies to raw model output. It closes the gap in the Claude-agent batch
# workflow (vlm_session_prompt.txt): each agent judges independently and is
# TRUSTED to satisfy R1..R11 itself, with no programmatic check. Running the
# written JSON through prompt_v2.py catches and deterministically fixes any
# rule violation (wrong risk/move for a grid_cell, missing open_side, etc.).
#
# Usage:
#   tools/vlm_rule_check.sh <out_dir> [--dry-run]
#
# Example:
#   tools/vlm_rule_check.sh log/e4b_20260902_105215.claude
set -euo pipefail

OUT_DIR="${1:?usage: vlm_rule_check.sh <out_dir> [--dry-run]}"
DRY_RUN=0
[[ "${2:-}" == "--dry-run" ]] && DRY_RUN=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHONPATH="$ROOT/src/drone_vlm" DRY_RUN="$DRY_RUN" python3 - "$OUT_DIR" <<'PY'
import json
import os
import sys
from pathlib import Path

from drone_vlm.prompt_v2 import repair_response

out_dir = Path(sys.argv[1])
dry_run = os.environ.get("DRY_RUN") == "1"

paths = sorted(out_dir.glob("frame_*.json"))
if not paths:
    print(f"{out_dir}: no frame_*.json files found.")
    sys.exit(0)

clean = fixed = broken = 0
for path in paths:
    try:
        obj = json.loads(path.read_text())
    except Exception as e:  # noqa: BLE001
        print(f"  {path.name}: UNREADABLE ({type(e).__name__}: {e})")
        broken += 1
        continue

    repaired, violations = repair_response(obj)
    control_violations = [v for v in violations if v != "R_SCENE"]

    if not violations:
        clean += 1
        continue

    fixed += 1
    tag = "advisory-only" if not control_violations else "REPAIRED"
    print(f"  {path.name}: {tag} [{','.join(violations)}]")
    if control_violations and not dry_run:
        path.write_text(json.dumps(repaired, ensure_ascii=False, indent=2))

print()
print(f"{out_dir}: {len(paths)} file(s) -- {clean} clean, {fixed} had violations"
      f"{' (dry-run: not written)' if dry_run else ''}, {broken} unreadable.")
PY
