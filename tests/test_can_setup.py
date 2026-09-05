"""Exercise the real shell wizard on fake sysfs and the GUI on harmless subprocesses."""

import queue
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "yam_udev_rules.sh"

# Source the actual script and substitute only host paths and hardware commands.
# No sudo, /etc writes, real USB probes, or CAN frames in these tests.
HARNESS = r'''
source "$1"
UDEV_FILE=$2
SYS_NET=$3
LOCK_FILE=$4
uname() { echo Linux; }
id() { echo 0; }
flock() { :; }
modprobe() { :; }
udevadm() {
    case "$1" in
        info) printf '    ATTRS{serial}=="%s"\n' "$(cat "$4/serial")" ;;
        control) [ "${FAIL_RELOAD:-0}" = 0 ] ;;
        settle) : ;;
        *) return 1 ;;
    esac
}
ip() { echo "$4: <UP> bitrate 1000000"; }
main --gui --target "$5"
'''


class Wizard:
    def __init__(self, tmp_path, targets="can_lead_l,can_left"):
        self.net = tmp_path / "net"
        self.net.mkdir()
        self.rules = tmp_path / "90-can.rules"
        self.original = (
            '# operator comment\n'
            'SUBSYSTEM=="net", ATTRS{serial}=="OLD", NAME="can_left"\n'
            'SUBSYSTEM=="net", ATTRS{serial}=="OTHER", NAME="can_right"\n'
        )
        self.rules.write_text(self.original)
        self.proc = subprocess.Popen(
            ["bash", "-c", HARNESS, "test-wizard", str(SCRIPT), str(self.rules),
             str(self.net), str(tmp_path / "lock"), targets],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        self.lines = queue.Queue()
        self.output = []
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            self.output.append(line)
            self.lines.put(line)
        self.lines.put(None)

    def prompt(self):
        while True:
            line = self.lines.get(timeout=5)
            assert line is not None, "".join(self.output)
            if line.startswith("@prompt "):
                return line

    def advance(self):
        self.proc.stdin.write("\n")
        self.proc.stdin.flush()

    def attach(self, iface, serial):
        path = self.net / iface
        path.mkdir()
        (path / "serial").write_text(serial)

    def close(self):
        if not self.proc.stdin.closed:
            self.proc.stdin.close()
        self.proc.wait(timeout=5)
        self.proc.stdout.close()


@pytest.fixture
def wizard(tmp_path):
    w = Wizard(tmp_path)
    yield w
    w.close()


def test_shell_binds_in_order_preserves_other_rules_and_backs_up(wizard):
    w = wizard
    assert "unplug" in w.prompt()
    # An unrelated adapter present at the baseline must not be mistaken for a target.
    w.attach("can_right", "OTHER")
    w.advance()
    assert "can_lead_l" in w.prompt()
    w.attach("can0", "NEW-LEADER")
    w.advance()
    assert "can_left" in w.prompt()
    w.attach("can1", "NEW-FOLLOWER")
    w.advance()
    assert "Install these bindings" in w.prompt()
    assert 'serial}=="OLD"' in w.rules.read_text()  # still intact before confirmation
    w.advance()
    assert "Unplug ALL" in w.prompt()
    w.advance()
    assert w.proc.wait(timeout=5) == 0, "".join(w.output)
    rules = w.rules.read_text()
    assert 'ATTRS{serial}=="NEW-LEADER", NAME="can_lead_l"' in rules
    assert 'ATTRS{serial}=="NEW-FOLLOWER", NAME="can_left"' in rules
    assert 'ATTRS{serial}=="OTHER", NAME="can_right"' in rules
    assert "OLD" not in rules
    assert "# operator comment" in rules
    assert rules.count("CAN device auto-configuration") == 1
    assert "gs_usb" in rules and "bitrate 1000000" in rules
    assert next(w.rules.parent.glob("90-can.rules.backup.*")).read_text() == w.original


def test_shell_refuses_ambiguous_devices_and_cancel_preserves_bindings(wizard):
    w = wizard
    w.prompt()
    w.advance()
    w.prompt()
    w.attach("can0", "ONE")
    w.attach("can1", "TWO")
    w.advance()
    assert "can_lead_l" in w.prompt()  # same prompt: no arbitrary first-match assignment
    assert "Detected 2 new adapters" in "".join(w.output)
    w.close()
    assert w.proc.returncode != 0
    assert 'ATTRS{serial}=="OLD", NAME="can_left"' in w.rules.read_text()
    assert 'ATTRS{serial}=="ONE"' not in w.rules.read_text()


@pytest.mark.parametrize("targets", ["", "can_left,", ",can_left", "can_left,can_left", "can_unused"])
def test_shell_invalid_targets_never_reach_host_setup(targets):
    proc = subprocess.run(
        ["bash", str(SCRIPT), "--target", targets], capture_output=True, text=True, timeout=5,
    )
    assert proc.returncode != 0
    assert "Linux robot host" not in proc.stderr


def wait_for(setup, status):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        state = setup.status()
        if state["status"] == status:
            return state
        time.sleep(0.01)
    pytest.fail(f"Expected {status}: {setup.status()}")


@pytest.fixture
def setup_client(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from yam_abc_reproduce.config import RobotConfig, StationConfig
    from yam_abc_reproduce.gui import can_setup
    from yam_abc_reproduce.gui.server import create_app

    script = tmp_path / "fake-setup.sh"
    script.write_text(
        "set -eu\nprintf '@prompt Plug in left leader.\\n'\nread -r\n"
        "printf 'Identified adapter.\\n@prompt Install bindings?\\n'\nread -r\n"
        "echo 'Setup complete.'\n"
    )
    monkeypatch.setattr(can_setup, "SCRIPT", script)
    monkeypatch.setattr(can_setup, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(can_setup.os, "geteuid", lambda: 0)
    cfg = StationConfig(robot=RobotConfig(type="mock"), save_root=str(tmp_path / "episodes"))
    with TestClient(create_app(cfg, mock=True)) as client:
        yield client


URL = "/api/maintenance/can-setup"


def test_gui_setup_prompts_survive_reconnect_and_reject_stale_answers(setup_client):
    c = setup_client
    setup = c.app.state.can_setup
    assert c.post(URL + "/start", json={"targets": ["can_left"]}).status_code == 200
    state = wait_for(setup, "waiting")
    assert c.get(URL).json()["prompt_id"] == state["prompt_id"]
    assert c.post(URL + "/start", json={"targets": ["can_left"]}).status_code == 409
    first_id = state["prompt_id"]
    assert c.post(URL + "/continue", json={"prompt_id": first_id}).status_code == 200
    state = wait_for(setup, "waiting")
    assert state["prompt_id"] > first_id
    assert c.post(URL + "/continue", json={"prompt_id": first_id}).status_code == 409
    assert c.post(URL + "/continue", json={"prompt_id": state["prompt_id"]}).status_code == 200
    assert wait_for(setup, "complete")["returncode"] == 0


def test_gui_setup_excludes_hardware_actions_until_cancelled(setup_client):
    c = setup_client
    setup = c.app.state.can_setup
    c.post(URL + "/start", json={"targets": ["can_left"]})
    wait_for(setup, "waiting")
    for path, body in [
        ("/api/collect/start-teleop", None), ("/api/deploy/start", {}),
        ("/api/maintenance/reset-can", None), ("/api/session/reset", None),
        ("/api/maintenance/zero-gello", {"side": "left"}),
    ]:
        assert c.post(path, json=body).status_code == 409
    assert c.post(URL + "/cancel").status_code == 200
    assert wait_for(setup, "cancelled")["active"] is False


def test_gui_setup_rejects_live_or_estopped_hardware(setup_client):
    c = setup_client
    session = c.app.state.session
    session.live = True
    assert c.post(URL + "/start", json={"targets": ["can_left"]}).status_code == 409
    session.live = False
    session.units = [object()]  # E-STOP may leave units open even though live is false.
    try:
        response = c.post(URL + "/start", json={"targets": ["can_left"]})
        assert response.status_code == 409
        assert "Reset Session" in response.json()["detail"]
    finally:
        session.units = []


@pytest.mark.parametrize("targets", [[], ["can_left", "can_left"], ["can_unknown"], ["can_left;true"]])
def test_gui_setup_validates_targets(setup_client, targets):
    assert setup_client.post(URL + "/start", json={"targets": targets}).status_code == 400


def test_gui_setup_handles_unsupported_host(setup_client, monkeypatch):
    from yam_abc_reproduce.gui import can_setup

    monkeypatch.setattr(can_setup, "sys", SimpleNamespace(platform="darwin"))
    assert setup_client.get(URL).json()["supported"] is False
    response = setup_client.post(URL + "/start", json={"targets": ["can_left"]})
    assert response.status_code == 409
    assert "Linux" in response.json()["detail"]


def test_gui_password_and_enter_drive_the_same_process(setup_client, monkeypatch):
    from yam_abc_reproduce.gui import can_setup

    monkeypatch.setattr(can_setup.os, "geteuid", lambda: 1000)
    real_popen = subprocess.Popen
    launched = []

    def fake_sudo(cmd, **kwargs):
        launched.append(cmd)
        # Authenticate without echoing the password, then run the actual fixture
        # script in the same process. Each GUI answer must reach its next read.
        return real_popen(
            ["bash", "-c", "printf '@sudo-password\\n'; read -r password; "
             'test "$password" = "do-not-log-this-password" || exit 1; exec "$@"',
             "fake-sudo", *cmd[6:]], **kwargs,
        )

    monkeypatch.setattr(can_setup.subprocess, "Popen", fake_sudo)
    c = setup_client
    password = "do-not-log-this-password"
    assert c.post(URL + "/start", json={"targets": ["can_left"], "password": password}).status_code == 200
    setup = c.app.state.can_setup
    state = wait_for(setup, "waiting")
    pid = setup._proc.pid
    assert launched[0][:6] == ["sudo", "-k", "-S", "-p", "@sudo-password\n", "--"]
    assert state["prompt"] == "Plug in left leader."
    assert password not in str(launched) + str(state)
    assert c.post(URL + "/continue", json={"prompt_id": state["prompt_id"]}).status_code == 200
    state = wait_for(setup, "waiting")
    assert state["prompt"] == "Install bindings?"
    assert setup._proc.pid == pid
    assert c.post(URL + "/continue", json={"prompt_id": state["prompt_id"]}).status_code == 200
    state = wait_for(setup, "complete")
    assert state["returncode"] == 0
    assert len(launched) == 1
    assert password not in str(state)


@pytest.mark.parametrize("needs_password", [False, True])
def test_gui_passwordless_and_failed_auth_do_not_advance_script(setup_client, monkeypatch, needs_password):
    from yam_abc_reproduce.gui import can_setup

    monkeypatch.setattr(can_setup.os, "geteuid", lambda: 1000)
    real_popen = subprocess.Popen
    # Either skip authentication (NOPASSWD), or reject the first password and retry.
    script = (
        "printf '@sudo-password\\n'; read -r password; "
        "printf 'Sorry, try again.\\n@sudo-password\\n'; read -r; exit 1"
        if needs_password else
        'printf "@prompt Ready.\\n"; read -r response; test -z "$response"'
    )
    monkeypatch.setattr(
        can_setup.subprocess, "Popen",
        lambda cmd, **kwargs: real_popen(["bash", "-c", script], **kwargs),
    )
    c = setup_client
    c.post(URL + "/start", json={"targets": ["can_left"], "password": "secret"})
    if needs_password:
        state = wait_for(c.app.state.can_setup, "failed")
        assert not state["active"]
        assert "secret" not in str(state)
    else:
        state = wait_for(c.app.state.can_setup, "waiting")
        c.post(URL + "/continue", json={"prompt_id": state["prompt_id"]})
        assert wait_for(c.app.state.can_setup, "complete")["returncode"] == 0
