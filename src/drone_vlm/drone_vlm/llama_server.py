"""LlamaServerManager — lifecycle wrapper around llama.cpp's llama-server.

Because inference happens in a SEPARATE PROCESS over HTTP, it does not hold the
Python GIL and does not stall rclpy (CLAUDE.md section 6.4).

There is NO in-process model object; memory is reclaimed by terminating the
server process (shutdown()), which is why the idle-unload path matters.

`requests` is imported lazily inside the methods that need it so this module
imports cleanly on a machine without `requests` installed. Install it from
drone_vlm/requirements.txt before using the real path.
"""
import shlex
import subprocess
import time
from typing import Callable, List, Optional


class LlamaServerCrashed(RuntimeError):
    pass


class LlamaServerManager:
    def __init__(self, logger, *, binary: str, model_path: str, mmproj_path: str,
                 model_name: str, port: int, n_gpu_layers: int, ctx_size: int,
                 startup_timeout: float, manage_server: bool,
                 extra_server_args: str = '',
                 server_log: Optional[str] = None) -> None:
        self._log = logger
        self.binary = binary
        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self.model_name = model_name
        self.port = int(port)
        self.n_gpu_layers = int(n_gpu_layers)
        self.ctx_size = int(ctx_size)
        self.startup_timeout = float(startup_timeout)
        self.manage_server = bool(manage_server)
        # Free-form pass-through of extra llama-server flags (e.g. "--reasoning off"
        # for Gemma, so reasoning tokens don't blow the max_tokens budget). Kept
        # generic so the node-managed invocation can match any working server.
        self.extra_server_args = extra_server_args or ''
        self.server_log = server_log or f'/tmp/llama_server_{self.port}.log'

        self.base_url = f'http://127.0.0.1:{self.port}'
        self.health_url = f'{self.base_url}/health'
        self.chat_url = f'{self.base_url}/v1/chat/completions'
        self._proc: Optional[subprocess.Popen] = None
        self._logf = None

    # ------------------------------------------------------------------ helpers
    def _health_ok(self, timeout: float = 2.0) -> bool:
        import requests  # lazy
        try:
            r = requests.get(self.health_url, timeout=(3.0, timeout))
            return r.status_code == 200
        except requests.RequestException:
            return False

    def _server_cmd(self) -> List[str]:
        cmd = [
            self.binary,
            '-m', self.model_path,
            '--port', str(self.port),
            '-ngl', str(self.n_gpu_layers),
            '-c', str(self.ctx_size),
            '--host', '127.0.0.1',
        ]
        if self.mmproj_path:
            cmd += ['--mmproj', self.mmproj_path]
        if self.extra_server_args:
            cmd += shlex.split(self.extra_server_args)
        return cmd

    # -------------------------------------------------------------------- API
    def ensure_up(self, stage_cb: Optional[Callable[[str], None]] = None) -> None:
        """Return once /health is 200. Start the server if managed and not up.

        stage_cb, if given, is called with 'starting_server' and 'waiting_health'
        so the action server can surface progress as feedback.
        """
        if self._health_ok():
            self._log.info('llama-server already healthy.')
            return
        if not self.manage_server:
            raise LlamaServerCrashed(
                'llama-server not healthy and manage_server=false '
                '(start it yourself, or set manage_server:=true).')

        if stage_cb:
            stage_cb('starting_server')
        self._log.info(f'Starting llama-server: {" ".join(self._server_cmd())}')
        self._log.info(f'llama-server stdout/stderr -> {self.server_log}')
        self._logf = open(self.server_log, 'wb')
        self._proc = subprocess.Popen(
            self._server_cmd(), stdout=self._logf, stderr=subprocess.STDOUT)

        if stage_cb:
            stage_cb('waiting_health')
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise LlamaServerCrashed(
                    f'llama-server exited immediately (code {self._proc.returncode}); '
                    f'see {self.server_log}.')
            if self._health_ok():
                self._log.info('llama-server is healthy.')
                return
            time.sleep(0.5)
        raise LlamaServerCrashed(
            f'llama-server did not become healthy within {self.startup_timeout}s; '
            f'see {self.server_log}.')

    def chat_raw(self, jpeg_b64: str, prompt: str, *, max_tokens: int, timeout,
                 response_format: Optional[dict] = None) -> dict:
        """POST one image+prompt to /v1/chat/completions; return the full JSON dict.

        Unlike chat(), this returns the whole server response (including the
        llama.cpp `timings` and `usage` blocks) and does NOT raise on a truncated
        response — so a benchmark can record timings even for a `length` finish.
        Raises only on HTTP/transport error.
        """
        import requests  # lazy
        payload = {
            'model': self.model_name,
            'temperature': 0,
            'max_tokens': int(max_tokens),
            'messages': [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': prompt},
                        {'type': 'image_url',
                         'image_url': {'url': f'data:image/jpeg;base64,{jpeg_b64}'}},
                    ],
                }
            ],
        }
        if response_format is not None:
            payload['response_format'] = response_format

        r = requests.post(self.chat_url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()

    def chat(self, jpeg_b64: str, prompt: str, *, max_tokens: int, timeout,
             response_format: Optional[dict] = None) -> str:
        """POST one image+prompt to /v1/chat/completions and return the text.

        Raises on HTTP error or if finish_reason == 'length' (truncated).
        """
        data = self.chat_raw(jpeg_b64, prompt, max_tokens=max_tokens,
                             timeout=timeout, response_format=response_format)
        choice = data['choices'][0]
        msg = choice['message']
        text = msg.get('content', '') or msg.get('reasoning_content', '')
        if choice.get('finish_reason') == 'length':
            raise RuntimeError(f'response truncated at max_tokens={max_tokens}')
        return text

    def shutdown(self) -> None:
        """terminate(), then kill() after 10 s. This is how memory is reclaimed."""
        if self._proc is None:
            return
        if self._proc.poll() is None:
            self._log.info('Shutting down llama-server (terminate).')
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                self._log.warn('llama-server did not exit in 10 s; killing.')
                self._proc.kill()
        self._proc = None
        if self._logf is not None:
            try:
                self._logf.close()
            except Exception:  # noqa: BLE001
                pass
            self._logf = None
