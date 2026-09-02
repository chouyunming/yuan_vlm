"""Prompt / schema for the drone obstacle-avoidance decider.

The decider replies with ONE JSON object:

  1. suggested_move : "left" | "right" | "forward" — a single high-level
     command. Downstream control code maps it onto its own velocity profile; a
     small quantised VLM ("gemma4") is far more reliable picking one of three
     labels than emitting self-consistent floats.

  2. The camera image is described to the model as a 3x3 grid. The model reports
     which grid cell the NEAREST obstacle occupies, and the hard rules pin down
     which suggested_move and risk each cell implies.

  3. open_side : "left" | "right" | "none" — reported ONLY when risk="high" (an
     obstacle sits in the flight line): the side with more free space to escape
     toward. When the path is blocked (a center cell), suggested_move steers
     toward this open_side so the drone escapes sideways instead of halting.

The grid (as the drone sees it, forward camera):

        col:   left        center        right
  row top    top_left    top_center    top_right      (far / high up)
  row mid    middle_left middle_center middle_right    (straight ahead)
  row bot    bottom_left bottom_center bottom_right    (near / low)
"""

import json
import re


# --------------------------------------------------------------------------- #
# Enumerations.
# --------------------------------------------------------------------------- #
_MOVES = ('left', 'right', 'forward')   # every suggested_move value
_SIDES = ('left', 'right', 'none')
_TYPES = ('tree', 'branch', 'unknown', 'none')
_RISKS = ('low', 'medium', 'high')
_CELLS = (
    'top_left', 'top_center', 'top_right',
    'middle_left', 'middle_center', 'middle_right',
    'bottom_left', 'bottom_center', 'bottom_right',
    'none',
)
# Escape side used to repair a high-risk reply whose open_side is missing.
_FALLBACK_SIDE = 'left'


# --------------------------------------------------------------------------- #
# Deterministic grid -> (risk, move) mapping. SINGLE source of truth, used both
# to describe the rules in the prompt and to repair a reply that violates them.
#
# HIGH risk = an obstacle in the flight line: ANY middle-row cell (middle_left /
# middle_center / middle_right, straight ahead) or bottom_center (near, dead
# ahead). Every high-risk cell forces a left/right avoidance: middle_left ->
# right, middle_right -> left, a blocked center cell -> open_side. top_center
# (far / high up) is MEDIUM -> keep forward. Corners and "none" are peripheral
# -> LOW -> forward.
# --------------------------------------------------------------------------- #
_HIGH_CELLS = ('middle_left', 'middle_center', 'middle_right', 'bottom_center')
_BLOCKED_CELLS = ('middle_center', 'bottom_center')   # path dead ahead -> escape sideways


def move_for_cell(cell: str, open_side: str = _FALLBACK_SIDE) -> str:
    """The suggested_move required when the nearest obstacle is in `cell`.
    A blocked center cell steers toward `open_side` (the reported escape side)."""
    if cell == 'middle_left':
        return 'right'                    # obstacle beside us on the left -> right
    if cell == 'middle_right':
        return 'left'                     # obstacle beside us on the right -> left
    if cell in _BLOCKED_CELLS:            # path dead ahead: escape sideways
        return open_side if open_side in ('left', 'right') else _FALLBACK_SIDE
    return 'forward'                      # top_center, corner, or none


def risk_for_cell(cell: str) -> str:
    """The collision risk implied by the nearest obstacle being in `cell`."""
    if cell in _HIGH_CELLS:
        return 'high'
    if cell == 'top_center':
        return 'medium'
    return 'low'                          # corner, or none


# --------------------------------------------------------------------------- #
# Prompt.
# --------------------------------------------------------------------------- #
_PROMPT_TEMPLATE = """You are the obstacle-avoidance decider for a drone with ONE forward camera. Reply with exactly ONE JSON object and NOTHING else.
Split the camera image into a 3x3 grid and name the cells:
        LEFT          CENTER          RIGHT
  TOP   top_left      top_center      top_right      (far / high up)
  MID   middle_left   middle_center   middle_right   (straight ahead)
  BOT   bottom_left   bottom_center   bottom_right   (near / low / close)
Find the NEAREST / most dangerous obstacle (tree, branch, wall, pole, ...), decide which ONE cell it mainly occupies, and report that cell.
FIELDS (report exactly these):
  scene            short free-text description of the view.
  obstacle_exists  true if any obstacle is visible, else false.
  grid_cell        the ONE cell holding the nearest obstacle, or "none" ONLY when obstacle_exists is false.
  obstacle_type    "tree"|"branch"|"unknown"|"none" ("none" ONLY when obstacle_exists is false).
  risk             collision risk for going forward: "low"|"medium"|"high".
  suggested_move   "left"|"right"|"forward". left/right=steer that way; forward=keep straight.
  open_side        escape direction: "left" or "right" WHENEVER risk="high" (the side with MORE free space); "none" whenever risk is not "high".
HARD RULES — satisfy ALL. Pick exactly ONE value per field; never output a list or an "a|b|c" string.
  R1. obstacle_exists false -> grid_cell="none", obstacle_type="none", risk="low", suggested_move="forward", open_side="none".
  R2. obstacle_exists true  -> grid_cell!="none" AND obstacle_type!="none".
  Grid cell -> suggested_move (an obstacle in the flight line forces a turn; far/high and corners keep going):
  R3. middle_left -> "right".
  R4. middle_right -> "left".
  R5. middle_center or bottom_center -> open_side ("left"/"right"): path blocked, escape sideways.
  R6. top_center -> "forward".
  R7. any CORNER cell or "none" -> "forward".
  Grid cell -> risk:
  R8.  any MIDDLE-ROW cell (middle_left/middle_center/middle_right) or bottom_center -> "high".
  R9.  top_center -> "medium".
  R10. any CORNER cell or "none" -> "low".
  R11. open_side is "left"/"right" IF AND ONLY IF risk="high" (required there, since R5 steers toward it); it is "none" whenever risk is not "high".
Reply with ONLY this JSON object (well-formed, rule-satisfying example):
{{"scene":"large tree blocking the path, grass open to the left","obstacle_exists":true,"grid_cell":"middle_center","obstacle_type":"tree","risk":"high","suggested_move":"left","open_side":"left"}}
"""


def build_prompt(memory: str) -> str:
    return _PROMPT_TEMPLATE.format(memory=memory or '(none)')


PROMPT = build_prompt('')


# --------------------------------------------------------------------------- #
# JSON extraction.
# --------------------------------------------------------------------------- #
def extract_json(text: str):
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise RuntimeError("no json found")
    return json.loads(m.group(0))


def has_placeholder_values(obj) -> bool:
    if isinstance(obj, str):
        return "|" in obj
    if isinstance(obj, dict):
        return any(has_placeholder_values(v) for v in obj.values())
    if isinstance(obj, list):
        return any(has_placeholder_values(v) for v in obj)
    return False


# --------------------------------------------------------------------------- #
# Deterministic rule enforcement.
#
# The json_schema response_format guarantees STRUCTURE (all keys present, each
# enum value legal). It cannot enforce the CROSS-FIELD LOGIC rules R1..R11 -- a
# small quantised VLM will occasionally emit a self-contradictory object.
# repair_response() rewrites such an object into one that satisfies every rule,
# deriving risk and suggested_move DIRECTLY from the reported grid cell (the
# precautionary source of truth) and gating open_side on risk="high".
# --------------------------------------------------------------------------- #
def repair_response(obj: dict):
    """Return (fixed_obj, violations) where fixed_obj satisfies every per-frame
    control rule R1..R11 and violations is the sorted list of rule codes the
    ORIGINAL object broke.
    "R_SCENE" is also flagged (advisory) when the free-text `scene` leaks an
    unresolved "a|b" placeholder; the control fields are still fully repaired.
    Non-destructive: works on a copy. Precautionary resolution."""
    o = dict(obj)
    bad = set()

    # Enum hygiene (schema should ensure this, but be defensive).
    if o.get('grid_cell') not in _CELLS:
        o['grid_cell'] = 'middle_center'
        bad.add('R1')
    if o.get('obstacle_type') not in _TYPES:
        o['obstacle_type'] = 'unknown'
        bad.add('R1')
    if o.get('risk') not in _RISKS:
        o['risk'] = 'medium'
        bad.add('R1')
    if o.get('suggested_move') not in _MOVES:
        o['suggested_move'] = 'forward'
        bad.add('R1')
    if o.get('open_side') not in _SIDES:
        o['open_side'] = 'none'
        bad.add('R1')

    ex = bool(o.get('obstacle_exists'))
    cell = o['grid_cell']
    typ = o['obstacle_type']

    # R1/R2: reconcile obstacle_exists with grid_cell/type. Precautionary -- an
    # obstacle is present if ANY signal says so.
    present = ex or cell != 'none' or typ != 'none'
    if present:
        if not ex:
            bad.add('R2')
        o['obstacle_exists'] = True
        if cell == 'none':
            o['grid_cell'] = cell = 'middle_center'
            bad.add('R2')
        if typ == 'none':
            o['obstacle_type'] = typ = 'unknown'
            bad.add('R2')
    else:
        o['obstacle_exists'] = False
        o['grid_cell'] = cell = 'none'
        o['obstacle_type'] = typ = 'none'

    # R8..R10: risk is fully determined by the grid cell.
    want_risk = risk_for_cell(cell)
    if o['risk'] != want_risk:
        bad.add('R8' if want_risk == 'high' else
                'R9' if want_risk == 'medium' else 'R10')
        o['risk'] = want_risk
    risk = o['risk']

    # R11 (biconditional): open_side is "left"/"right" IF AND ONLY IF risk is
    # "high" -- required there because R5 steers a blocked center cell toward it
    # -- and "none" otherwise. Resolved BEFORE the move for that reason.
    open_side = o['open_side']
    if risk == 'high':
        if open_side not in ('left', 'right'):   # model failed to pick a side
            bad.add('R11')
            open_side = _FALLBACK_SIDE
    else:
        if open_side != 'none':
            bad.add('R11')
        open_side = 'none'
    o['open_side'] = open_side

    # R3..R7: suggested_move is fully determined by the grid cell (a blocked
    # center cell steers toward open_side).
    want_move = move_for_cell(cell, open_side)
    if o['suggested_move'] != want_move:
        if cell == 'middle_left':
            bad.add('R3')
        elif cell == 'middle_right':
            bad.add('R4')
        elif cell in _BLOCKED_CELLS:      # middle_center / bottom_center: blocked
            bad.add('R5')
        elif cell == 'top_center':
            bad.add('R6')
        else:
            bad.add('R7')
        o['suggested_move'] = want_move

    # R_SCENE: the free-text `scene` must not leak an unresolved "a|b"
    # placeholder -- the ONE placeholder case the enum handling above does not
    # already cover. Advisory: flagged only, not rewritten (scene is not a
    # control field, so it does not affect the movement command).
    if has_placeholder_values(o.get('scene')):
        bad.add('R_SCENE')

    return o, sorted(bad)


def check_response(obj: dict):
    """Return the sorted list of rule codes `obj` violates, without modifying it."""
    return repair_response(obj)[1]


RESPONSE_FORMAT = {
    'type': 'json_schema',
    'json_schema': {
        'name': 'scene_command',
        'strict': True,
        'schema': {
            'type': 'object',
            'properties': {
                'scene': {'type': 'string', 'maxLength': 120},
                'obstacle_exists': {'type': 'boolean'},
                'grid_cell': {'enum': list(_CELLS)},
                'obstacle_type': {'enum': list(_TYPES)},
                'risk': {'enum': list(_RISKS)},
                'suggested_move': {'enum': list(_MOVES)},
                'open_side': {'enum': list(_SIDES)},
            },
            'required': ['scene', 'obstacle_exists', 'grid_cell', 'obstacle_type',
                         'risk', 'suggested_move', 'open_side'],
        },
    },
}
