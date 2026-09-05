"""Run the interactive YAM udev script, with explicit browser acknowledgements."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

from .jobs import _signal_group

TARGETS = ("can_lead_l", "can_lead_r", "can_left", "can_right")
SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "yam_udev_rules.sh"
_SUDO_PROMPT = "@sudo-password"


class CanSetup:
    def __init__(self):
        self.lock = threading.RLock()
        self._proc: subprocess.Popen | None = None
        self._status = "idle"
        self._logs: list[str] = []
        self._prompt: str | None = None
        self._prompt_id = 0
        self._returncode: int | None = None

    @property
    def active(self) -> bool:
        with self.lock:
            return self._status in {"running", "waiting", "cancelling"}

    def status(self) -> dict:
        # Cached state only: the regular status feed never probes USB/CAN.
        with self.lock:
            return {
                "supported": sys.platform == "linux",
                "status": self._status,
                "active": self.active,
                "prompt": self._prompt,
                "prompt_id": self._prompt_id,
                "returncode": self._returncode,
                "logs": list(self._logs),
            }

    def start(self, targets: list[str], password: str = "") -> dict:
        with self.lock:
            if self.active:
                raise RuntimeError("CAN setup is already running; finish or cancel it first.")
            if not targets or len(set(targets)) != len(targets) or set(targets) - set(TARGETS):
                raise ValueError(f"Select unique CAN targets from: {', '.join(TARGETS)}")
            if "\n" in password or "\r" in password:
                raise ValueError("The sudo password must be a single line.")
            if sys.platform != "linux":
                raise RuntimeError("CAN setup requires a Linux robot host.")
            if not SCRIPT.is_file():
                raise RuntimeError(f"Setup script missing: {SCRIPT}. Run the GUI from this repo.")
            cmd = ["bash", str(SCRIPT), "--gui", "--target", ",".join(targets)]
            use_password = bool(password) and os.geteuid() != 0
            if os.geteuid() != 0:
                # Wait for sudo's explicit prompt before sending a password.
                # NOPASSWD skips authentication even with -k: pre-filling stdin
                # would otherwise answer the script's first hardware prompt.
                auth = ["-k", "-S", "-p", _SUDO_PROMPT + "\n"] if use_password else ["-n"]
                cmd = ["sudo", *auth, "--", *cmd]
            proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1,
                start_new_session=True,
                env={**os.environ, "LC_ALL": "C"},
            )
            self._proc = proc
            self._status = "running"
            self._prompt = None
            self._returncode = None
            self._logs = ["Starting YAM CAN setup…"]
            # The password is never stored on this object or included in argv/logs.
            threading.Thread(
                target=self._pump, args=(proc, password if use_password else ""), daemon=True,
            ).start()
            return self.status()

    def _pump(self, proc: subprocess.Popen, password: str) -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            if line == _SUDO_PROMPT:
                # One attempt per launch. Close input on a retry instead of
                # leaving sudo waiting indefinitely for another password.
                with self.lock:
                    try:
                        if password and proc.stdin and not proc.stdin.closed:
                            proc.stdin.write(password + "\n")
                            proc.stdin.flush()
                        elif proc.stdin:
                            proc.stdin.close()
                    except (OSError, ValueError):
                        pass
                    password = ""
                continue
            with self.lock:
                if line.startswith("@prompt "):
                    password = ""  # sudo may have allowed the command without a password
                    if self._status != "cancelling":
                        self._prompt = line.removeprefix("@prompt ")
                        self._prompt_id += 1
                        self._status = "waiting"
                        self._logs.append(self._prompt)
                else:
                    self._logs.append(line)
                del self._logs[:-500]
        proc.wait()
        proc.stdout.close()
        if proc.stdin and not proc.stdin.closed:
            try:
                proc.stdin.close()
            except OSError:
                pass
        with self.lock:
            cancelled = self._status == "cancelling"
            self._returncode = proc.returncode
            self._status = "cancelled" if cancelled else ("complete" if proc.returncode == 0 else "failed")
            self._prompt = None
            if self._status == "failed":
                self._logs.append(
                    "Check the error above. If sudo needs authentication, enter the robot-host "
                    "sudo password and retry, or run: sudo bash scripts/yam_udev_rules.sh"
                )

    def advance(self, prompt_id: int) -> dict:
        with self.lock:
            if self._status != "waiting" or prompt_id != self._prompt_id:
                raise RuntimeError("This setup prompt has changed. Wait for the current prompt.")
            assert self._proc is not None and self._proc.stdin is not None
            try:
                self._proc.stdin.write("\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, ValueError) as exc:
                raise RuntimeError("Setup has exited; check its log and start again.") from exc
            self._status = "running"
            self._prompt = None
            return self.status()

    def cancel(self) -> dict:
        with self.lock:
            if not self.active or self._proc is None:
                return self.status()
            self._status = "cancelling"
            self._prompt = None
            proc = self._proc
            if proc.stdin and not proc.stdin.closed:
                try:
                    proc.stdin.close()
                except OSError:
                    pass
            _signal_group(proc, signal.SIGTERM)
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            # Keep the operation active until its output reader sees exit.
            _signal_group(proc, signal.SIGKILL)
        return self.status()
