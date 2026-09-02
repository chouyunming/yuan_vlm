"""Logging & run-recording helpers for the VLM benchmark.

This module owns everything that records a run to disk — both the continuous
resource log and the end-of-run summary:

  * ``TegrastatsLogger`` spawns ``tegrastats --logfile ...`` for the whole test
    session so CPU/GPU/RAM/power/temp are recorded continuously (CLAUDE.md:
    tegrastats covers the whole interval, not just inference). The parse/
    aggregate helpers are vendored from
    ~/nvidia/gemma4_orin/scripts/tegrastats_log.py so this stays self-contained
    inside the ROS package (no import across the workspace boundary).
  * the stats / summary helpers (``build_summary``, ``write_run_summary``,
    ``render_summary_txt``) turn a list of per-frame records into
    ``summary.{json,txt}``.

NOTE: parse_line requires the leading `MM-DD-YYYY HH:MM:SS` timestamp that
tegrastats writes in --logfile mode on this Jetson; lines without it are skipped.
"""
import json
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Optional

TIMESTAMP_RE = re.compile(r"^(\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2})")
RAM_RE = re.compile(r"RAM (\d+)/(\d+)MB")
SWAP_RE = re.compile(r"SWAP (\d+)/(\d+)MB")
CPU_RE = re.compile(r"CPU \[([^\]]+)\]")
CPU_CORE_RE = re.compile(r"(\d+)%@\d+")
GPU_RE = re.compile(r"GR3D_FREQ (\d+)%")
# Orin-family boards report VDD_IN; some Jetson boards report POM_5V_IN instead.
POWER_RE = re.compile(r"(?:VDD_IN|POM_5V_IN) (\d+)mW")
TEMP_RES = [
    re.compile(r"tj@([\d.]+)C"),
    re.compile(r"cpu@([\d.]+)C"),
    re.compile(r"gpu@([\d.]+)C"),
]


def parse_line(line: str):
    """Parse one tegrastats output line into a flat dict, or None if unparseable."""
    ts_m = TIMESTAMP_RE.match(line)
    if not ts_m:
        return None
    timestamp = datetime.strptime(ts_m.group(1), "%m-%d-%Y %H:%M:%S")

    ram_m = RAM_RE.search(line)
    ram_used_mb = int(ram_m.group(1)) if ram_m else None
    ram_total_mb = int(ram_m.group(2)) if ram_m else None

    swap_m = SWAP_RE.search(line)
    swap_used_mb = int(swap_m.group(1)) if swap_m else None

    cpu_avg_pct = None
    cpu_m = CPU_RE.search(line)
    if cpu_m:
        cores = [int(p) for p in CPU_CORE_RE.findall(cpu_m.group(1))]
        if cores:
            cpu_avg_pct = sum(cores) / len(cores)

    gpu_m = GPU_RE.search(line)
    gpu_pct = int(gpu_m.group(1)) if gpu_m else None

    power_m = POWER_RE.search(line)
    power_mw = int(power_m.group(1)) if power_m else None

    temp_c = None
    for temp_re in TEMP_RES:
        temp_m = temp_re.search(line)
        if temp_m:
            temp_c = float(temp_m.group(1))
            break

    return {
        "timestamp": timestamp,
        "ram_used_mb": ram_used_mb,
        "ram_total_mb": ram_total_mb,
        "swap_used_mb": swap_used_mb,
        "cpu_avg_pct": cpu_avg_pct,
        "gpu_pct": gpu_pct,
        "power_mw": power_mw,
        "temp_c": temp_c,
    }


def load_log(log_path) -> List[dict]:
    """Parse a tegrastats log file into a list of sample dicts, in file order."""
    samples = []
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            sample = parse_line(line)
            if sample is not None:
                samples.append(sample)
    return samples


def _stat(samples, key):
    vals = [s[key] for s in samples if s.get(key) is not None]
    if not vals:
        return None
    return {"mean": sum(vals) / len(vals), "min": min(vals), "max": max(vals)}


def aggregate(samples: List[dict]) -> Optional[dict]:
    """Aggregate parsed tegrastats samples into mean/min/max per metric."""
    if not samples:
        return None
    ts = [s["timestamp"] for s in samples if s.get("timestamp")]
    duration_sec = (max(ts) - min(ts)).total_seconds() if len(ts) >= 2 else None
    return {
        "n_samples": len(samples),
        "duration_sec": duration_sec,
        "gpu_pct": _stat(samples, "gpu_pct"),
        "cpu_avg_pct": _stat(samples, "cpu_avg_pct"),
        "power_mw": _stat(samples, "power_mw"),
        "ram_used_mb": _stat(samples, "ram_used_mb"),
        "ram_total_mb": samples[-1].get("ram_total_mb"),
        "temp_c": _stat(samples, "temp_c"),
    }


class TegrastatsLogger:
    """Spawn `tegrastats --logfile <path>` for the session; stop() ends it."""

    def __init__(self, log_path, interval_ms: int = 500, logger=None) -> None:
        self.log_path = Path(log_path)
        self.interval_ms = int(interval_ms)
        self._log = logger
        self._proc: Optional[subprocess.Popen] = None

    def _info(self, msg: str) -> None:
        if self._log is not None:
            self._log.info(msg)
        else:
            print(msg)

    def _warn(self, msg: str) -> None:
        if self._log is not None:
            self._log.warn(msg)
        else:
            print(msg)

    def start(self) -> None:
        if shutil.which("tegrastats") is None:
            self._warn("tegrastats not found on PATH; skipping resource logging.")
            return
        try:
            self._proc = subprocess.Popen(
                ["tegrastats", "--interval", str(self.interval_ms),
                 "--logfile", str(self.log_path)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._info(f"tegrastats -> {self.log_path} (every {self.interval_ms} ms)")
        except Exception as e:  # noqa: BLE001
            self._warn(f"failed to start tegrastats: {type(e).__name__}: {e}")
            self._proc = None

    def stop(self) -> None:
        if self._proc is None:
            return
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None
        self._info(f"tegrastats stopped ({self.log_path}).")


# --------------------------------------------------------------------------- stats
def _percentile(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    k = (len(s) - 1) * (q / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def _stats(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return {
        'n': len(vals),
        'mean': sum(vals) / len(vals),
        'min': min(vals),
        'max': max(vals),
        'p50': _percentile(vals, 50),
        'p95': _percentile(vals, 95),
    }


def fmt(v):
    """Format a millisecond/percent value for the human-readable summary."""
    return f'{v:.0f}' if isinstance(v, (int, float)) else '?'


# --------------------------------------------------------------- summary build
def build_summary(cfg, records, *, run_dir, frames_seen, exposure_rejects,
                  source_label, model_file, tegra=None, logger=None) -> dict:
    """Turn per-frame records + run metadata into the summary dict written to
    ``summary.json``. Aggregates the tegrastats log if one was recorded."""
    recs = list(records)  # snapshot: worker may still append on Ctrl-C
    ok = [r for r in recs if r.get('ok')]
    summary = {
        'model': cfg.model,
        'model_file': model_file,
        'response_format': cfg.response_format,
        'max_tokens': cfg.max_tokens,
        'source': source_label,
        'frames_inferred': len(recs),
        'frames_ok': len(ok),
        'frames_seen': frames_seen,
        'exposure_rejects': exposure_rejects,
        'placeholder_echoes': sum(1 for r in recs if r.get('placeholder')),
        'rule_violations': sum(1 for r in recs if r.get('violations')),
        'truncated': sum(1 for r in recs if r.get('truncated')),
        'errors': sum(1 for r in recs if r.get('error')),
        'latency_ms': {
            'wall': _stats([r.get('wall_ms') for r in recs]),
            'prompt_eval': _stats([r.get('prompt_ms') for r in ok]),
            'generation': _stats([r.get('predicted_ms') for r in ok]),
            'total': _stats([r.get('total_ms') for r in ok]),
        },
        'throughput_tok_s': {
            'prompt_eval': _stats([r.get('prompt_per_second') for r in ok]),
            'generation': _stats([r.get('predicted_per_second') for r in ok]),
        },
        'tokens': {
            'prompt': _stats([r.get('prompt_tokens') for r in ok]),
            'completion': _stats([r.get('completion_tokens') for r in ok]),
        },
        'tegrastats': None,
        'per_frame': recs,
    }
    if tegra is not None:
        try:
            summary['tegrastats'] = aggregate(load_log(run_dir / 'tegrastats.log'))
        except Exception as e:  # noqa: BLE001
            if logger is not None:
                logger.warn(f'tegrastats parse failed: {e}')
    return summary


def write_run_summary(cfg, records, *, run_dir, frames_seen, exposure_rejects,
                      source_label, model_file, tegra=None, logger=None) -> None:
    """Build the summary and write ``summary.{json,txt}`` into ``run_dir``."""
    summary = build_summary(
        cfg, records, run_dir=run_dir, frames_seen=frames_seen,
        exposure_rejects=exposure_rejects, source_label=source_label,
        model_file=model_file, tegra=tegra, logger=logger)
    (run_dir / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding='utf-8')
    (run_dir / 'summary.txt').write_text(
        render_summary_txt(summary), encoding='utf-8')


# ----------------------------------------------------------------- summary render
def _lat_row(label, st):
    if not st:
        return f'  {label:<14} (no data)'
    return (f'  {label:<14} p50={fmt(st["p50"])}ms  p95={fmt(st["p95"])}ms  '
            f'mean={fmt(st["mean"])}ms  min={fmt(st["min"])}ms  max={fmt(st["max"])}ms')


def _tegra_row(label, st, unit):
    if not st:
        return f'  {label:<14} (no data)'
    return f'  {label:<14} mean={st["mean"]:.1f}{unit}  max={st["max"]:.1f}{unit}'


def render_summary_txt(s: dict) -> str:
    lines = []
    lines.append(f'VLM inference benchmark — {s["model"]} ({s["model_file"]})')
    lines.append('=' * 62)
    lines.append(f'source            : {s["source"]}')
    lines.append(f'response_format   : {s["response_format"]}   max_tokens: {s["max_tokens"]}')
    lines.append(f'frames inferred   : {s["frames_inferred"]}  (ok={s["frames_ok"]}, '
                 f'seen={s["frames_seen"]}, exposure_rejects={s["exposure_rejects"]})')
    lines.append(f'placeholder echoes: {s["placeholder_echoes"]}   '
                 f'rule violations (repaired): {s.get("rule_violations", 0)}   '
                 f'truncated: {s["truncated"]}   errors: {s["errors"]}')
    lines.append('')
    lines.append('LATENCY (llama.cpp timings, over ok frames; wall over all)')
    lines.append('-' * 62)
    lat = s['latency_ms']
    lines.append(_lat_row('prompt eval', lat['prompt_eval']))
    lines.append(_lat_row('generation', lat['generation']))
    lines.append(_lat_row('total', lat['total']))
    lines.append(_lat_row('wall (client)', lat['wall']))
    tp = s['throughput_tok_s']
    if tp['prompt_eval'] or tp['generation']:
        pe = tp['prompt_eval']
        ge = tp['generation']
        lines.append('')
        lines.append('THROUGHPUT (tok/s, mean)')
        lines.append('-' * 62)
        lines.append(f'  prompt eval    {pe["mean"]:.1f}' if pe else '  prompt eval    (no data)')
        lines.append(f'  generation     {ge["mean"]:.1f}' if ge else '  generation     (no data)')
    tk = s['tokens']
    if tk['completion']:
        c = tk['completion']
        p = tk['prompt']
        lines.append('')
        lines.append('TOKENS')
        lines.append('-' * 62)
        if p:
            lines.append(f'  prompt         {p["mean"]:.0f} (fixed)')
        lines.append(f'  completion     mean={c["mean"]:.1f}  min={c["min"]:.0f}  max={c["max"]:.0f}')
    tg = s.get('tegrastats')
    if tg:
        lines.append('')
        lines.append(f'RESOURCES (tegrastats, whole session, '
                     f'{tg.get("n_samples")} samples over '
                     f'{fmt(tg.get("duration_sec"))}s)')
        lines.append('-' * 62)
        lines.append(_tegra_row('GPU util', tg.get('gpu_pct'), '%'))
        lines.append(_tegra_row('CPU avg', tg.get('cpu_avg_pct'), '%'))
        if tg.get('power_mw'):
            pw = tg['power_mw']
            lines.append(f'  {"power":<14} mean={pw["mean"]/1000.0:.2f}W  max={pw["max"]/1000.0:.2f}W')
        if tg.get('ram_used_mb'):
            rr = tg['ram_used_mb']
            tot = tg.get('ram_total_mb')
            lines.append(f'  {"RAM used":<14} mean={rr["mean"]/1024.0:.2f}GB  '
                         f'max={rr["max"]/1024.0:.2f}GB'
                         + (f'  / {tot/1024.0:.1f}GB' if tot else ''))
        lines.append(_tegra_row('temp', tg.get('temp_c'), 'C'))
    lines.append('')
    return '\n'.join(lines)
