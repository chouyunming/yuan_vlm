#!/usr/bin/env python3
"""Select the next batch of frames for Claude-driven VLM inference.

This does NOT run any model. It only figures out which frames in a
--frame-dir still need a JSON output in {frame-dir}.claude/, starting at
--start-frame, up to --batch-size frames. Claude reads its printed list
and does the actual per-frame inference (Read image -> Write JSON).

Usage:
    python3 tools/vlm_batch_select.py --frame-dir log/e4b_20260902_105215 --start-frame 20
    python3 tools/vlm_batch_select.py --frame-dir log/e4b_20260902_105215 --start-frame 20 --batch-size 50
"""
import argparse
import json
import re
from pathlib import Path

FRAME_RE = re.compile(r"frame_(\d{6})\.jpg$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame-dir", required=True, help="folder containing frame_XXXXXX.jpg")
    ap.add_argument("--start-frame", type=int, required=True, help="first frame index to consider")
    ap.add_argument("--batch-size", type=int, default=50)
    args = ap.parse_args()

    frame_dir = Path(args.frame_dir)
    out_dir = frame_dir.parent / f"{frame_dir.name}.claude"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_frames = {}
    for p in frame_dir.glob("frame_*.jpg"):
        m = FRAME_RE.search(p.name)
        if m:
            all_frames[int(m.group(1))] = p

    end_frame = args.start_frame + args.batch_size
    indices_in_range = sorted(i for i in all_frames if args.start_frame <= i < end_frame)

    pending = []
    already_done = []
    for i in indices_in_range:
        out_path = out_dir / f"frame_{i:06d}.json"
        if out_path.exists():
            already_done.append(i)
        else:
            pending.append(all_frames[i])

    max_index = max(all_frames) if all_frames else -1
    next_start = end_frame if end_frame <= max_index else None

    result = {
        "frame_dir": str(frame_dir),
        "out_dir": str(out_dir),
        "requested_range": [args.start_frame, end_frame - 1],
        "total_frames_in_dir": len(all_frames),
        "already_done_count": len(already_done),
        "pending_count": len(pending),
        "pending_frames": [str(p) for p in pending],
        "next_start_frame": next_start,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
