#!/usr/bin/env python3
"""vlm_test — ROS-FREE VLM (Gemma) inference-time benchmark.

Runs with a plain Python interpreter (NO ROS, NO rclpy):

    python -m drone_vlm.vlm_test --model e4b --video ~/videos/test.MOV
    python -m drone_vlm.vlm_test --model e4b --image ~/frames/frame_000282.jpg

It reads frames straight off disk (a .MOV via OpenCV, or a single .jpg) and runs
Gemma back-to-back on each one — no camera, no topic, no publish/subscribe. This
is the reproducible, off-drone benchmark path.

The ROS path — live-OAK topic subscription and the publish/subscribe mock
camera — lives in ``vlm_node.py`` and runs under ``ros2 run drone_vlm
vlm_node``. Both share the inference engine ``VlmBenchmark`` (defined in
``vlm_node.py``); the driver here subclasses it and adds only the off-disk
frame sources.

Two input modes (mutually exclusive):
  * ``--video foo.MOV`` — decode-and-infer: exhaustive, in-order, single pass.
    Every decoded frame is inferred exactly once. ``--num-frames`` defaults to
    the clip's total frame count; if ``--num-frames N`` is below the total, N
    frame positions are sampled uniformly across the whole clip.
  * ``--image foo.jpg`` — run inference once on a .jpg and exit.

For each inferred frame it records llama.cpp ``timings`` (prompt_ms /
predicted_ms / tok-s) plus client wall-clock, saves the frame and responses
under ``~/yuan_vlm/log/<model>_<timestamp>/``, runs tegrastats across the whole
session, and writes ``summary.{json,txt}`` at the end (all of that recording
logic lives in ``bench_log.py``).

It ATTACHES to an already-running llama-server; it never launches one. Start it
first, e.g. ``start_e4b_server.sh``.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2

from drone_vlm.vlm_node import VlmBenchmark, add_common_args


# --------------------------------------------------------- standalone driver
class VlmBenchmarkRunner(VlmBenchmark):
    """Standalone (ROS-free) driver: single-image and decode-and-infer video.

    Reads frames directly off disk and infers them synchronously, in order —
    there is no publish/subscribe race, so every decoded frame is inferred
    exactly once."""

    def __init__(self, cfg: argparse.Namespace, logger=None) -> None:
        super().__init__(cfg, logger)
        self._cap = None
        self._video_total_frames = 0
        if cfg.video and not cfg.image:
            self._source_suffix = '(decode-and-infer)'
            self._open_video(cfg)

    def _open_video(self, cfg: argparse.Namespace) -> None:
        """Open the capture — no publisher/timer. The run loop reads frames from
        ``self._cap`` directly, in order."""
        path = str(Path(cfg.video).expanduser())
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            self.log.error(f'cannot open video: {path}')
            self.done.set()
            return
        fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._video_total_frames = total
        if cfg.num_frames is None:
            if total > 0:
                cfg.num_frames = total
            else:
                # Some containers/codecs don't report a frame count; fall back
                # to an effectively unbounded cap and let cap.read() returning
                # False (end of stream) be the real stop condition.
                cfg.num_frames = sys.maxsize
                self.log.warn(
                    f'{path}: could not determine frame count; will decode '
                    'until end-of-stream.')
        self.log.info(
            f'decode-and-infer: {path} ({total or "?"} frames @ {fps:.1f} fps) '
            f'-- synchronous, in-order, exhaustive pass, num_frames={cfg.num_frames}')

    def run(self) -> None:
        """Drive the benchmark to completion, then write the summary."""
        try:
            if self.cfg.image:
                self._run_image()
            else:
                self._run_decode_and_infer()
        finally:
            try:
                if self._cap is not None:
                    self._cap.release()
            except Exception:  # noqa: BLE001
                pass
            self._finish()

    def _run_image(self) -> None:
        """Single-image mode: infer once on a .jpg read straight off disk."""
        path = Path(self.cfg.image).expanduser()
        jpeg = path.read_bytes()
        with self._lock:
            self._frames_seen += 1
        self.records.append(self._infer_one(0, jpeg))

    def _run_decode_and_infer(self) -> None:
        """Decode-and-infer dispatch:
          * num_frames >= total (or total unknown) -> sequential exhaustive
            pass, every decoded frame inferred once, in order.
          * num_frames < total -> uniformly SAMPLE that many frame positions
            across the whole clip (not just the first num_frames frames)."""
        total = self._video_total_frames
        if total > 0 and self.cfg.num_frames < total:
            self._decode_and_infer_sampled(total)
        else:
            self._decode_and_infer_sequential()

    def _decode_and_infer_sequential(self) -> None:
        idx = 0
        while not self.done.is_set() and idx < self.cfg.num_frames:
            ret, frame = self._cap.read()
            if not ret:
                self.log.info(f'video exhausted after {idx} frames; stopping.')
                break
            self._encode_and_infer(idx, frame)
            idx += 1
            if self.cfg.interval_sec > 0:
                time.sleep(self.cfg.interval_sec)

    def _decode_and_infer_sampled(self, total: int) -> None:
        n = self.cfg.num_frames
        positions = [round(i * (total - 1) / max(n - 1, 1)) for i in range(n)]
        self.log.info(
            f'decode-and-infer: sampling {n} frames evenly across '
            f'{total} total frames (positions {positions[0]}..{positions[-1]}).')
        for idx, pos in enumerate(positions):
            if self.done.is_set():
                break
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = self._cap.read()
            if not ret:
                self.log.warn(f'failed to read frame at position {pos}; skipping.')
                continue
            self._encode_and_infer(idx, frame)
            if self.cfg.interval_sec > 0:
                time.sleep(self.cfg.interval_sec)


# -------------------------------------------------------------------------- args
def _parse_args(argv):
    p = argparse.ArgumentParser(
        prog='vlm_test',
        description='ROS-free VLM (Gemma) inference-time benchmark over a .MOV '
                    '(decode-and-infer) or a single .jpg.')
    add_common_args(p)
    p.add_argument('--video', default='~/yuan_vlm/videos/test.MOV',
                   help='decode-and-infer this .MOV, exhaustive in-order pass '
                        '(default source if --image is not given)')
    p.add_argument('--image', default=None,
                   help='single-image mode: run inference once on this .jpg and '
                        'exit -- no video needed.')
    cfg = p.parse_args(argv)
    if cfg.image:
        if cfg.num_frames is not None and cfg.num_frames != 1:
            p.error('--image always infers exactly one frame; drop --num-frames')
        cfg.num_frames = 1
        cfg.video = None
    # else: --num-frames left None is resolved once the video is opened
    # (_open_video), against its actual total frame count.
    return cfg


def main() -> None:
    cfg = _parse_args(sys.argv[1:])
    runner = VlmBenchmarkRunner(cfg)
    if not runner.start():
        sys.exit(1)
    try:
        runner.run()
    except KeyboardInterrupt:
        runner.log.info('interrupted; finalising...')
        runner.stop()


if __name__ == '__main__':
    main()
