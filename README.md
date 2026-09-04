# yuan_vlm

A single-package ROS 2 workspace for **`drone_vlm`** — obstacle-avoidance inference
with the Gemma-4 VLM served by `llama-server`, benchmarked on a **Jetson Orin NX**
(JetPack 6 / R36.4, Ubuntu 22.04, Python 3.10, ROS 2 Humble).

```
yuan_vlm/
├── models/               # Gemma-4 GGUF + mmproj (~9.7 GB, gitignored)
├── videos/               # test clips (gitignored)
├── build/llama.cpp/      # native CUDA llama.cpp build (build-cuda/bin/llama-server)
├── start_e4b_server.sh   # launch the E4B llama-server (native build)
├── start_e2b_server.sh   # launch the E2B llama-server (native build)
├── run_all_videos.sh     # run vlm_test over every .mp4 in a directory
└── src/drone_vlm/        # the ROS 2 package
```

`drone_vlm` provides two entry points that share one inference engine (`VlmBenchmark`)
and the same output format:

| | `vlm_test` (no ROS) | `vlm_node` (ROS) |
|---|---|---|
| Run with | `python -m drone_vlm.vlm_test …` | `ros2 run drone_vlm vlm_node …` |
| Frame source | `--video` (off-disk decode) / `--image` (single `.jpg`) | `--image-topic` (live subscription), or `--video` as a mock camera |
| Output | `~/yuan_vlm/log/<model>_<timestamp>/` | same |

Both **attach** to an already-running `llama-server` — they never launch one.

---

## Setup

**System packages (apt, not pip):**

```bash
sudo apt install ros-humble-ros-base python3-colcon-common-extensions
sudo apt install python3-opencv    # JetPack/CUDA build — do NOT pip install opencv-python
```

`tegrastats` ships with JetPack (`nvidia-l4t-tools`); no install needed.

**Python packages (pip):** only `requests` is required — `numpy`/OpenCV come from
JetPack, don't override them.

```bash
pip install -r src/drone_vlm/requirements.txt
```

**Native llama.cpp (CUDA) build** — the leak-free server path. `build/` is
gitignored, so clone llama.cpp there and build once:

```bash
mkdir -p ~/yuan_vlm/build && cd ~/yuan_vlm/build
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 \
      -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_SERVER=ON
cmake --build build-cuda --target llama-server -j"$(nproc)"
```

Produces `build/llama.cpp/build-cuda/bin/llama-server`, run directly by the
`start_*.sh` scripts (no docker, no sudo).

**Models** under `~/yuan_vlm/models/`:

| model | GGUF | mmproj |
|---|---|---|
| e4b | `gemma-4-E4B-it-Q4_K_M.gguf` | `mmproj-gemma-4-E4B-it-Q8_0.gguf` |
| e2b | `google_gemma-4-E2B-it-Q4_K_M.gguf` | `mmproj-google_gemma-4-E2B-it-f16.gguf` |

> The `start_*.sh` scripts mount `/home/anderson/yuan_vlm/models`. If your home
> directory differs, edit the `-v` line in both scripts.

## Build

```bash
cd ~/yuan_vlm
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select drone_vlm
source install/setup.bash
```

`--symlink-install` lets you edit the `.py` files without rebuilding; re-run
`colcon build` only after changing `setup.py` or `package.xml`.

## Run

Start a server first (separate terminal, stays in the foreground):

```bash
./start_e4b_server.sh      # or ./start_e2b_server.sh (smaller/faster)
```

Then run a benchmark:

```bash
# vlm_test — ROS-free, off-disk (no rclpy, no camera)
python -m drone_vlm.vlm_test --model e4b --video ~/yuan_vlm/videos/mission_record_forest.mp4
python -m drone_vlm.vlm_test --model e4b --image path/to/frame.jpg     # single frame

# vlm_node — ROS image topic (source ROS + the workspace first)
source /opt/ros/humble/setup.bash && source ~/yuan_vlm/install/setup.bash
ros2 run drone_vlm vlm_node --model e4b --image-topic front_camera/image/compressed   # live camera
ros2 run drone_vlm vlm_node --model e4b --video ~/yuan_vlm/videos/mission_record_forest.mp4  # mock camera
```

Pass `--help` for all flags (`--num-frames`, `--port`, `--out-dir`,
`--response-format`, `--no-tegrastats`, …).

## Output

Each run writes `~/yuan_vlm/log/<model>_<timestamp>/`:

```
frame_000000.jpg               # frames actually inferred
frame_000000.json              # repaired, rule-compliant decision
frame_000000.raw_response.json # raw llama-server response (timings, usage)
tegrastats.log                 # whole-session CPU/GPU/RAM/power/temp
summary.json / summary.txt      # p50/p95/mean latency + tegrastats aggregates
```

## Notes

- **`vlm_node --image-topic` needs a publisher** on the topic. With no camera, use
  `vlm_test --video` (off-disk) or `vlm_node --video` (mock camera that republishes a
  clip onto the topic at its fps).
- `vlm_node` publishes each decided step on **`/vlm/result`** as JSON
  `{"command_id": N, "suggested_move": "left|right|forward"}` (fire-and-forget; the
  flight-controller handshake is deferred to a later redesign).
- One model per server run; switch by restarting with the other `start_*.sh`.
