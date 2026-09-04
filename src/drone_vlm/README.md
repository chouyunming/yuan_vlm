# drone_vlm — VLM (Gemma) inference benchmark: `vlm_test` + `vlm_node`

Two executables, split by whether they need ROS. Both measure **actual VLM inference
time** — per-frame llama.cpp `timings` (`prompt_ms` / `predicted_ms` / tok-s) + client
wall-clock — and share one inference engine (`VlmBenchmark`) and output format.

- **`vlm_test`** — **ROS-free**. Reads frames straight off disk: `--video` (decode-and-
  infer a `.MOV`) or `--image` (a single `.jpg`). Runs with a plain interpreter, no rclpy.
- **`vlm_node`** — the **ROS** path: `--image-topic` subscribes to a live OAK camera and
  infers the freshest frame each cycle. `vlm_node` subclasses `VlmBenchmark` and adds the
  rclpy I/O; the run loop infers the latest frame, never inside a ROS callback. Each
  decided step is published on `/vlm/result` as `{"command_id", "suggested_move"}`
  (fire-and-forget — the flight-controller handshake is deferred to a later redesign).

This README covers bringing up a **fresh Jetson Orin board** to run them.
Reference board: **JetPack R36.4 / JetPack 6**, Ubuntu 22.04, Python 3.10, ROS 2 Humble.

Both share the same output, written to `~/yuan_vlm/log/<model>_<timestamp>/`:

- **only frames actually inferred**: `frame_NNNNNN.{jpg,json,raw_response.json}`.
- **`tegrastats`** run continuously over the whole session (CPU/GPU/RAM/power/temp).
- `summary.{json,txt}` with **p50/p95/mean** latency + tegrastats aggregates.

`vlm_node`'s frame source is either a **live OAK** topic (`--image-topic`, needs a
publisher — see [Notes](#notes)) or a **mock camera** (`--video foo.MOV` republishes the
clip onto the topic at its fps, no hardware).

---

## Prerequisites

### 1. System packages (apt — not pip)

```bash
# ROS 2 Humble (provides rclpy, sensor_msgs) + colcon
sudo apt install ros-humble-ros-base python3-colcon-common-extensions

# OpenCV (cv2) — use the JetPack/apt build (reference board: 5.0.0, CUDA-enabled).
# Do NOT `pip install opencv-python` on Jetson — it fetches a 4.x CPU wheel that
# shadows the JetPack build.
sudo apt install python3-opencv
```

`tegrastats` ships with JetPack (`nvidia-l4t-tools`); no install needed. It is used by
`bench_log.py`.

### 2. Python packages (pip)

```bash
pip install -r src/drone_vlm/requirements.txt
```

`requests` is the only pip dependency `vlm_test` truly needs. `numpy` is normally already
present via ROS 2 / JetPack (reference board: 2.2.6) — **don't override the system one**.

> This is intentionally **not** a `pip freeze`. On Jetson, numpy / OpenCV / torch come from
> JetPack / NVIDIA wheels; reinstalling them from PyPI breaks the CUDA/ABI build
> (see the workspace `CLAUDE.md` §10).

### 3. Native llama.cpp (CUDA) + Gemma server

Every mode needs the server running (the `--video` / mock-camera paths just don't need a
camera). The Gemma `llama-server` is a **native CUDA build of llama.cpp** — no docker,
no sudo — which the `start_*.sh` scripts launch directly. Build it once:

```bash
cd ~/yuan_vlm/build/llama.cpp
cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 \
      -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_SERVER=ON
cmake --build build-cuda --target llama-server -j"$(nproc)"
```

This yields `build/llama.cpp/build-cuda/bin/llama-server`; `start_e4b_server.sh` sets
`LD_LIBRARY_PATH` and runs it. (`-DCMAKE_CUDA_ARCHITECTURES=87` targets the Orin NX.)

Models — Gemma-4 GGUF + mmproj under `~/yuan_vlm/models/`:

| model | GGUF | mmproj |
|---|---|---|
| e4b | `gemma-4-E4B-it-Q4_K_M.gguf` | `mmproj-gemma-4-E4B-it-Q8_0.gguf` |
| e2b | `google_gemma-4-E2B-it-Q4_K_M.gguf` | `mmproj-google_gemma-4-E2B-it-f16.gguf` |

---

## Build

```bash
cd ~/yuan_vlm
colcon build --symlink-install --packages-select drone_vlm
source install/setup.bash
```

`--symlink-install` lets you edit `vlm_test.py` / `vlm_node.py` / `bench_log.py` without
rebuilding. Re-run `colcon build` after editing `setup.py` or `package.xml`.

## Run

Start the server first (separate terminal):

```bash
~/yuan_vlm/start_e4b_server.sh
```

Then pick a node:

```bash
# vlm_test — ROS-free, off-disk (no rclpy, no camera)
python -m drone_vlm.vlm_test --model e4b --video ~/yuan_vlm/videos/test.MOV   # decode-and-infer
python -m drone_vlm.vlm_test --model e4b --image ~/yuan_vlm/frames/frame_000282.jpg # single frame

# vlm_node — ROS path (source ROS + the workspace first)
source /opt/ros/humble/setup.bash && source ~/yuan_vlm/install/setup.bash
ros2 run drone_vlm vlm_node --model e4b --image-topic front_camera/image/compressed  # live OAK
ros2 run drone_vlm vlm_node --model e4b --video ~/yuan_vlm/videos/mission_record_forest.mp4  # mock camera
```

Use `--help` on either (`python -m drone_vlm.vlm_test --help`,
`ros2 run drone_vlm vlm_node --help`) for all flags (`--num-frames`, `--port`, `--out-dir`,
`--response-format`, `--no-tegrastats`, …).

## Output

`~/yuan_vlm/log/<model>_<timestamp>/`:

```
frame_000000.jpg              # only frames actually inferred
frame_000000.json             # repaired, rule-compliant decision JSON
frame_000000.raw_response.json# raw llama-server response (has `timings`, `usage`)
...
tegrastats.log                # whole-session CPU/GPU/RAM/power/temp
summary.json                  # structured p50/p95/mean + tegrastats aggregates
summary.txt                   # human-readable summary
```

---

## Notes

- **`vlm_node --image-topic` needs a publisher on `front_image/compressed`.** With no
  camera, use `vlm_test --video` for off-disk benchmarks, or `vlm_node --video` as a mock
  camera, or add a standalone OAK publisher.
- Both nodes **attach** to an already-running server; they never launch `llama-server`
  themselves. Start it via `start_e4b_server.sh` first.
- Model per run is fixed; switch by restarting the server with the other `start_*` script.
