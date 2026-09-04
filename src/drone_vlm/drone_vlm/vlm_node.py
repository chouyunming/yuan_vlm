#!/usr/bin/env python3
"""vlm_node — VLM (Gemma) inference engine + the ROS image-topic path.

``VlmBenchmark`` is the ROS-free inference engine: it attaches to an
already-running llama-server, infers one JPEG at a time, records llama.cpp
timings, writes per-frame artifacts, and delegates the run summary to
:mod:`drone_vlm.bench_log`. ``vlm_test.VlmBenchmarkRunner`` (off-disk decode)
and ``VlmNode`` (ROS image topic) both subclass it.

``VlmNode`` adds the rclpy I/O: ``--image-topic TOPIC`` subscribes to a live
image topic (e.g. an OAK camera on ``front_camera/image/compressed``) and infers the
freshest frame each cycle, dropping frames that arrive mid-inference. Pass
``--video foo.MOV`` to run as a MOCK CAMERA that republishes a clip onto
``--image-topic``. Each decided step is published on ``/vlm/result``
(fire-and-forget, no ack).

It ATTACHES to an already-running llama-server (start it first, e.g.
``start_e4b_server.sh``); it never launches one.
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from drone_vlm.bench_log import TegrastatsLogger, fmt, write_run_summary
from drone_vlm.llama_server import LlamaServerManager
from drone_vlm.prompt_v2 import (PROMPT, RESPONSE_FORMAT, extract_json,
                                 has_placeholder_values, repair_response)

# rclpy is only needed by VlmNode; import it lazily so the engine (and
# vlm_test, which imports it) works under a plain interpreter with no rclpy.
try:
    import rclpy
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.utilities import remove_ros_args
    from sensor_msgs.msg import CompressedImage
    from std_msgs.msg import String
except ImportError:  # ROS-free environment: only the engine is usable.
    rclpy = None


MODEL_CHOICES = {
    'e2b': 'google_gemma-4-E2B-it-Q4_K_M.gguf',
    'e4b': 'gemma-4-E4B-it-Q4_K_M.gguf',
}

# Topic each decided per-frame step is published on (fire-and-forget, no ack).
RESULT_TOPIC = '/vlm/result'


# ------------------------------------------------------------------------ logging
class StdLogger:
    """Minimal stand-in for a ROS node logger (``.info/.warn/.error``) backed by
    the stdlib ``logging`` module, so the shared engine can log identically
    whether it runs standalone or inside a ROS node."""

    def __init__(self, name: str = 'vlm') -> None:
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s',
                datefmt='%H:%M:%S')
        self._l = logging.getLogger(name)

    def info(self, msg):
        self._l.info(msg)

    def warn(self, msg):
        self._l.warning(msg)

    warning = warn

    def error(self, msg):
        self._l.error(msg)

    def debug(self, msg):
        self._l.debug(msg)


# ---------------------------------------------------------------------- engine
class VlmBenchmark:
    """ROS-free inference engine: attach to llama-server, infer one JPEG at a
    time, record timings, write per-frame artifacts and a run summary.

    It owns everything that does NOT need ROS. ``vlm_test.VlmBenchmarkRunner``
    (off-disk decode) and ``VlmNode`` (ROS image topic) subclass it."""

    def __init__(self, cfg: argparse.Namespace, logger=None) -> None:
        self.cfg = cfg
        self.log = logger or StdLogger()
        self.done = threading.Event()
        self._finished = False
        self._finish_lock = threading.Lock()

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = Path(cfg.out_dir).expanduser() / f'{cfg.model}_{ts}'
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log.info(f'run dir: {self.run_dir}')

        self.response_format = {
            'json_schema': RESPONSE_FORMAT,
            'json_object': {'type': 'json_object'},
            'none': None,
        }[cfg.response_format]

        # Attach-only server manager (manage_server=False -> ensure_up() just
        # health-checks; it will NOT try to launch a bare llama-server).
        self.manager = LlamaServerManager(
            self.log, binary='llama-server', model_path='', mmproj_path='',
            model_name=MODEL_CHOICES[cfg.model], port=cfg.port, n_gpu_layers=99,
            ctx_size=1024, startup_timeout=30.0, manage_server=False,
            extra_server_args='')

        self._lock = threading.Lock()
        self._frames_seen = 0
        self._exposure_rejects = 0
        # Suffix appended to the source label in the summary (e.g. the mock
        # camera vs decode-and-infer). Set by the driver/subclass.
        self._source_suffix = ''

        self.tegra = None
        if not cfg.no_tegrastats:
            self.tegra = TegrastatsLogger(
                self.run_dir / 'tegrastats.log',
                interval_ms=cfg.tegra_interval_ms, logger=self.log)

        self.records = []

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> bool:
        """Health-check the server and start tegrastats. Returns False (and sets
        ``done``) if the server is not reachable."""
        try:
            self.manager.ensure_up()
        except Exception as e:  # noqa: BLE001
            self.log.error(
                f'llama-server not reachable on 127.0.0.1:{self.cfg.port}: '
                f'{type(e).__name__}: {e}. Start it first '
                f'(e.g. start_{self.cfg.model}_server.sh) and retry.')
            self.done.set()
            return False
        if self.tegra is not None:
            self.tegra.start()
        self.log.info(
            f'benchmark started: model={self.cfg.model} '
            f'target_frames={self.cfg.num_frames} source={self._source_label()}')
        return True

    def _source_label(self) -> str:
        if getattr(self.cfg, 'image', None):
            base = f'image:{self.cfg.image}'
        elif getattr(self.cfg, 'video', None):
            base = f'video:{self.cfg.video}'
        else:
            base = f'topic:{getattr(self.cfg, "image_topic", "?")}'
        return base + self._source_suffix

    def stop(self) -> None:
        self._finish()

    def _publish_step(self, obj: dict, stem: str) -> None:
        """Emit one decided step. No-op here; ``VlmNode`` publishes it on
        ``/vlm/result``."""
        return None

    # -------------------------------------------------------------------- infer
    def _encode_and_infer(self, idx: int, frame: np.ndarray) -> None:
        ok, buf = cv2.imencode(
            '.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.cfg.jpeg_quality])
        if not ok:
            return
        jpeg = np.asarray(buf).tobytes()
        with self._lock:
            self._frames_seen += 1
        self.records.append(self._infer_one(idx, jpeg))

    def _infer_one(self, idx: int, jpeg: bytes) -> dict:
        stem = f'frame_{idx:06d}'
        jpeg_b64 = base64.b64encode(jpeg).decode()
        rec = {'idx': idx, 'frame': f'{stem}.jpg'}
        t0 = time.perf_counter()
        try:
            data = self.manager.chat_raw(
                jpeg_b64, PROMPT, max_tokens=self.cfg.max_tokens,
                timeout=(10, self.cfg.infer_timeout),
                response_format=self.response_format)
            wall_ms = (time.perf_counter() - t0) * 1000.0

            (self.run_dir / f'{stem}.raw_response.json').write_text(
                json.dumps(data, ensure_ascii=False), encoding='utf-8')
            (self.run_dir / f'{stem}.jpg').write_bytes(jpeg)

            choice = data['choices'][0]
            msg = choice['message']
            text = msg.get('content', '') or msg.get('reasoning_content', '')
            truncated = choice.get('finish_reason') == 'length'
            timings = data.get('timings', {}) or {}
            usage = data.get('usage', {}) or {}

            placeholder = False
            perr = None
            violations = None
            try:
                obj = extract_json(text)
                placeholder = has_placeholder_values(obj)
                # Repair into a rule-compliant object; the written .json is the
                # control-ready version, raw model text stays in
                # {stem}.raw_response.json.
                obj, violations = repair_response(obj)
                (self.run_dir / f'{stem}.json').write_text(
                    json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
                self._publish_step(obj, stem)
            except Exception as e:  # noqa: BLE001
                perr = f'{type(e).__name__}: {e}'

            prompt_ms = timings.get('prompt_ms')
            predicted_ms = timings.get('predicted_ms')
            total_ms = None
            if prompt_ms is not None and predicted_ms is not None:
                total_ms = prompt_ms + predicted_ms

            rec.update({
                'ok': perr is None and not truncated,
                'wall_ms': wall_ms,
                'prompt_ms': prompt_ms,
                'prompt_n': timings.get('prompt_n'),
                'prompt_per_second': timings.get('prompt_per_second'),
                'predicted_ms': predicted_ms,
                'predicted_n': timings.get('predicted_n'),
                'predicted_per_second': timings.get('predicted_per_second'),
                'total_ms': total_ms,
                'prompt_tokens': usage.get('prompt_tokens'),
                'completion_tokens': usage.get('completion_tokens'),
                'truncated': truncated,
                'placeholder': placeholder,
                'violations': violations,
                'parse_error': perr,
            })
            self.log.info(
                f'[{idx + 1}/{self.cfg.num_frames}] {stem}: '
                f'prompt={fmt(prompt_ms)}ms gen={fmt(predicted_ms)}ms '
                f'total={fmt(total_ms)}ms wall={wall_ms:.0f}ms'
                f'{" PLACEHOLDER" if placeholder else ""}'
                f'{" TRUNC" if truncated else ""}'
                f'{" FIXED:" + ",".join(violations) if violations else ""}'
                f'{" ERR:" + perr if perr else ""}')
        except Exception as e:  # noqa: BLE001 — record the failure, keep going
            wall_ms = (time.perf_counter() - t0) * 1000.0
            rec.update({'ok': False, 'wall_ms': wall_ms,
                        'error': f'{type(e).__name__}: {e}'})
            try:
                (self.run_dir / f'{stem}.jpg').write_bytes(jpeg)
            except Exception:  # noqa: BLE001
                pass
            self.log.warn(f'[{idx + 1}] {stem} inference failed: {rec["error"]}')
        return rec

    # --------------------------------------------------------------------- finish
    def _finish(self) -> None:
        with self._finish_lock:
            if self._finished:
                self.done.set()
                return
            self._finished = True
        if self.tegra is not None:
            self.tegra.stop()
        if self.records:
            write_run_summary(
                self.cfg, self.records, run_dir=self.run_dir,
                frames_seen=self._frames_seen,
                exposure_rejects=self._exposure_rejects,
                source_label=self._source_label(),
                model_file=MODEL_CHOICES[self.cfg.model],
                tegra=self.tegra, logger=self.log)
        self.log.info(
            f'done: {len(self.records)} frames inferred -> {self.run_dir}')
        self.done.set()


# -------------------------------------------------------------------------- args
def add_common_args(p: argparse.ArgumentParser) -> None:
    """Inference/output args shared by the standalone benchmark and vlm_node."""
    p.add_argument('--model', choices=['e2b', 'e4b'], default='e4b')
    p.add_argument('--num-frames', type=int, default=None,
                   help='stop after this many frames are actually inferred '
                        "(default: video's total frame count for --video, 1 for "
                        '--image)')
    p.add_argument('--port', type=int, default=8080,
                   help='llama-server port (attach only; server must already run)')
    p.add_argument('--out-dir', default='~/yuan_vlm/log',
                   help='run dirs are created under here as <model>_<timestamp>')
    p.add_argument('--interval-sec', type=float, default=0.0,
                   help='sleep between inferences (0 = back-to-back)')
    p.add_argument('--jpeg-quality', type=int, default=90,
                   help='JPEG quality for re-encoding decoded/published frames')
    p.add_argument('--tegra-interval-ms', type=int, default=500)
    p.add_argument('--max-tokens', type=int, default=1024)
    p.add_argument('--infer-timeout', type=float, default=120.0)
    p.add_argument('--response-format', choices=['json_schema', 'json_object', 'none'],
                   default='json_schema')
    p.add_argument('--no-tegrastats', action='store_true')


# --------------------------------------------------------------------------- #
# ROS-aware engine: VlmBenchmark + rclpy I/O (live image-topic subscription).
# --------------------------------------------------------------------------- #
class VlmNode(VlmBenchmark):
    """VlmBenchmark subclass that sources frames from a live ROS topic.

    It owns an rclpy node (composition, not ``Node`` subclassing — the base
    engine is plain Python) plus a background executor. The main thread runs the
    benchmark loop, pulling the freshest subscribed frame and inferring it."""

    def __init__(self, cfg: argparse.Namespace) -> None:
        # rclpy must already be initialised (see main()).
        self.node = rclpy.create_node('vlm_node')
        # Route the shared engine's logging through the ROS logger.
        super().__init__(cfg, logger=self.node.get_logger())
        self._cb = ReentrantCallbackGroup()

        # Freshest-frame buffer (live topic or mock-camera republish).
        self._latest = None
        self._latest_lock = threading.Lock()
        self._new_frame = threading.Event()

        # Fire-and-forget publisher for the decided step (RELIABLE, depth 10).
        # command_id is a strictly increasing per-step counter.
        self._result_pub = self.node.create_publisher(String, RESULT_TOPIC, 10)
        self._command_id = 0

        self._make_subscriptions(cfg)
        self._make_mock_camera(cfg)

        if cfg.image_topic:
            self._source_suffix = '(live-topic, freshest-frame)'
        self.log.info(f'publishing decided steps -> {RESULT_TOPIC} (no ack)')

        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self.node)
        self._spin_thread = threading.Thread(
            target=self._executor.spin, daemon=True)

    # ------------------------------------------------------------- ROS wiring
    def _make_subscriptions(self, cfg: argparse.Namespace) -> None:
        """Subscribe to the image topic as sensor_msgs/CompressedImage (what the
        OAK camera and the mock camera both publish)."""
        self.node.create_subscription(
            CompressedImage, cfg.image_topic, self._on_compressed_image,
            qos_profile_sensor_data, callback_group=self._cb)
        self.log.info(f'subscribed to compressed image topic: {cfg.image_topic}')

    def _make_mock_camera(self, cfg: argparse.Namespace) -> None:
        """Optional: republish a local clip onto --image-topic so the live path
        can be exercised with no camera. Decoding runs in one dedicated thread
        (:meth:`_mock_loop`), not a ROS timer: this OpenCV/FFmpeg build aborts
        if a VideoCapture is opened on one thread and read on another."""
        self._mock_pub = None
        self._mock_thread = None
        self._mock_path = None
        self._mock_fps = 30.0
        if not cfg.video:
            return
        path = str(Path(cfg.video).expanduser())
        # Probe once here just for fps / to fail fast on a bad path, then release
        # immediately; the decode capture is opened inside the decode thread.
        probe = cv2.VideoCapture(path)
        if not probe.isOpened():
            self.log.error(f'--video mock camera: cannot open {path}')
            self.done.set()
            return
        self._mock_fps = probe.get(cv2.CAP_PROP_FPS) or 30.0
        probe.release()
        self._mock_path = path
        self._mock_pub = self.node.create_publisher(
            CompressedImage, cfg.image_topic, qos_profile_sensor_data)
        self._mock_thread = threading.Thread(target=self._mock_loop, daemon=True)
        self.log.info(
            f'mock camera: republishing {path} @ {self._mock_fps:.1f} fps '
            f'-> {cfg.image_topic}')

    # ------------------------------------------------------------- callbacks
    def _stash(self, frame: np.ndarray) -> None:
        with self._latest_lock:
            self._latest = frame
        self._new_frame.set()

    def _on_compressed_image(self, msg: CompressedImage) -> None:
        frame = cv2.imdecode(
            np.frombuffer(bytes(msg.data), dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is not None:
            self._stash(frame)

    def _mock_loop(self) -> None:
        """Dedicated decode thread: own the capture for its whole life, decode
        the clip in a loop, and republish each frame on --image-topic at the
        clip's fps until the run ends."""
        cap = cv2.VideoCapture(self._mock_path)
        if not cap.isOpened():
            self.log.error(f'--video mock camera: cannot open {self._mock_path}')
            self.done.set()
            return
        period = 1.0 / max(self._mock_fps, 1.0)
        try:
            while not self.done.is_set():
                ret, frame = cap.read()
                if not ret:                       # loop the clip
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret:
                        return
                ok, buf = cv2.imencode(
                    '.jpg', frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), self.cfg.jpeg_quality])
                if ok:
                    msg = CompressedImage()
                    msg.header.stamp = self.node.get_clock().now().to_msg()
                    msg.format = 'jpeg'
                    msg.data = np.asarray(buf).tobytes()
                    self._mock_pub.publish(msg)
                time.sleep(period)
        finally:
            cap.release()

    def _publish_step(self, obj: dict, stem: str) -> None:
        """Publish {command_id, suggested_move, obstacle_exists} on /vlm/result
        as a JSON String. command_id strictly increases so a subscriber can
        order/deduplicate."""
        self._command_id += 1
        msg = String()
        msg.data = json.dumps(
            {'command_id': self._command_id,
             'suggested_move': str(obj.get('suggested_move', '')),
             'obstacle_exists': bool(obj.get('obstacle_exists', False))})
        self._result_pub.publish(msg)

    # -------------------------------------------------------------- run loop
    def run(self) -> None:
        """Spin the executor in the background and infer the freshest frame each
        cycle until num_frames is reached or the topic goes quiet."""
        self._spin_thread.start()
        if self._mock_thread is not None:
            self._mock_thread.start()
        try:
            idx = 0
            while not self.done.is_set() and idx < self.cfg.num_frames:
                if not self._new_frame.wait(self.cfg.frame_wait_timeout):
                    self.log.warn(
                        f'no new frame on {self.cfg.image_topic} for '
                        f'{self.cfg.frame_wait_timeout}s; still waiting...')
                    continue
                self._new_frame.clear()
                with self._latest_lock:
                    frame = self._latest
                    self._latest = None
                if frame is None:
                    continue
                self._encode_and_infer(idx, frame)
                idx += 1
                if self.cfg.interval_sec > 0:
                    time.sleep(self.cfg.interval_sec)
        finally:
            self._finish()

    def shutdown(self) -> None:
        # Stop the mock camera first and wait for it to exit, so it can't publish
        # on a node that the lines below are about to destroy (teardown race).
        self.done.set()
        if self._mock_thread is not None:
            self._mock_thread.join(timeout=2.0)
        try:
            self._executor.shutdown(timeout_sec=1.0)
        except Exception:  # noqa: BLE001
            pass
        try:
            self.node.destroy_node()
        except Exception:  # noqa: BLE001
            pass


# -------------------------------------------------------------------------- args
def _parse_args(argv):
    p = argparse.ArgumentParser(
        prog='vlm_node',
        description='ROS VLM (Gemma) path: live image-topic subscription '
                    '(optionally fed by a --video mock camera).')
    add_common_args(p)
    p.add_argument('--image-topic', default='front_camera/image/compressed',
                   help='CompressedImage topic to subscribe to (live OAK or '
                        'mock camera)')
    p.add_argument('--video', default=None,
                   help='optional MOCK CAMERA: republish this clip onto '
                        '--image-topic so the live path runs without hardware')
    p.add_argument('--frame-wait-timeout', type=float, default=15.0,
                   help='warn if no new frame arrives on the topic for this long')
    cfg = p.parse_args(argv)
    # Topic mode has no natural frame count; default to 1000 inferred.
    if cfg.num_frames is None:
        cfg.num_frames = 1000
    cfg.image = None    # this node never runs the ROS-free single-image path
    return cfg


def main(args=None) -> None:
    if rclpy is None:
        raise RuntimeError(
            'vlm_node requires rclpy/ROS 2, which is not importable in this '
            'environment. Use `python -m drone_vlm.vlm_test` for the ROS-free '
            'off-drone benchmark.')
    rclpy.init(args=args)
    argv = remove_ros_args(sys.argv)[1:]
    cfg = _parse_args(argv)
    node = VlmNode(cfg)
    try:
        if not node.start():
            sys.exit(1)
        node.run()
    except KeyboardInterrupt:
        node.log.info('interrupted; finalising...')
        node.stop()
    finally:
        node.shutdown()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
