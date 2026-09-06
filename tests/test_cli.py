"""Exercise CLI boundaries without connecting to hardware or loading policy weights."""

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import tyro

from scripts.calibrate_gello import CalibrateGelloArgs
from scripts.check_can_usb_quality import CheckCanUsbQualityArgs
from yam_abc_reproduce.cli import ConvertArgs, GuiArgs
from yam_abc_reproduce.deploy.merge_abc_lora import MergeAbcLoraArgs
from yam_abc_reproduce.deploy.run import DeployArgs

ROOT = Path(__file__).resolve().parents[1]
ENTRY_POINTS = [
    *(f"yam_abc_reproduce.cli:{name}" for name in ("cameras", "teleop", "convert", "viz", "gui")),
    "yam_abc_reproduce.deploy.run:deploy",
    "yam_abc_reproduce.doctor:main",
    "yam_abc_reproduce.deploy.merge_abc_lora:main",
    *(
        f"yam_abc_reproduce.deploy.servers.{name}_server:main"
        for name in ("abc", "molmoact", "openpi")
    ),
    *(
        f"scripts.{name}:main"
        for name in (
            "calibrate_gello",
            "check_can_usb_quality",
            "compute_abc_norm_stats",
            "monitor_can_reply_latency",
            "setup_can_udev",
            "yam_episodes_to_abc_mcap",
        )
    ),
]


@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
def test_help_works_without_optional_dependencies(entry_point):
    module, function = entry_point.split(":")
    code = f"""
import builtins
import importlib
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'fastapi', 'uvicorn', 'i2rt', 'mcap', 'abc_minimal',
                              'openpi', 'pyrealsense2', 'rerun', 'cv2'):
        raise ModuleNotFoundError(name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
getattr(importlib.import_module({module!r}), {function!r})()
"""
    result = subprocess.run(
        [sys.executable, "-c", code, "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "--help" in result.stdout


def test_positionals_boolean_flags_and_interface_lists():
    assert tyro.cli(ConvertArgs, args=["my episodes", "--to", "abc"]).src == "my episodes"
    calibration = tyro.cli(CalibrateGelloArgs, args=["monitor", "--controller", "right"])
    assert calibration.command == "monitor" and calibration.controller == "right"
    merge = tyro.cli(MergeAbcLoraArgs, args=["lora.pt", "merged.pt"])
    assert (merge.src, merge.dst) == ("lora.pt", "merged.pt")
    assert tyro.cli(GuiArgs, args=["--mock"]).mock is True
    interfaces = tyro.cli(CheckCanUsbQualityArgs, args=["--interfaces", "can0", "can1"])
    assert interfaces.interfaces == ["can0", "can1"]
    deploy = tyro.cli(
        DeployArgs,
        args=[
            "--host",
            "localhost",
            "--port",
            "8000",
            "--prompt",
            "pick up the block",
            "--rtc",
            "--home-pose",
            "",
            "--max-joint-speed",
            "0.8",
        ],
    )
    assert deploy.rtc and deploy.home_pose == "" and deploy.max_joint_speed == 0.8


@pytest.mark.parametrize(
    ("args_type", "argv"),
    [
        (ConvertArgs, []),
        (ConvertArgs, ["episodes", "--to", "invalid"]),
        (CalibrateGelloArgs, ["invalid"]),
        (CalibrateGelloArgs, ["monitor", "--controller", "invalid"]),
        (MergeAbcLoraArgs, ["lora.pt"]),
        (DeployArgs, ["--host", "localhost", "--port", "8000"]),
        (GuiArgs, ["--port", "not-a-number"]),
        (GuiArgs, ["--unknown-option"]),
    ],
)
def test_invalid_arguments_exit_with_usage_error(args_type, argv):
    with pytest.raises(SystemExit) as exc:
        tyro.cli(args_type, args=argv, console_outputs=False)
    assert exc.value.code == 2


@pytest.mark.parametrize("module_name", ["check_can_usb_quality", "monitor_can_reply_latency"])
def test_empty_interface_list_is_rejected_before_sampling(module_name, monkeypatch, capsys):
    module = importlib.import_module(f"scripts.{module_name}")
    monkeypatch.setattr(sys, "argv", [module_name, "--interfaces"])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2
    assert "at least one interface" in capsys.readouterr().err


@pytest.mark.parametrize("option", ["--seconds", "--interval"])
def test_can_monitor_rejects_nonpositive_duration(option, monkeypatch, capsys):
    from scripts.check_can_usb_quality import main

    monkeypatch.setattr(sys, "argv", ["check_can_usb_quality", option, "0"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert "must be positive" in capsys.readouterr().err


def test_doctor_no_listen_and_json(monkeypatch, capsys):
    from yam_abc_reproduce import doctor

    def fake_run(config, do_listen):
        assert config == Path("custom.yaml") and do_listen is False
        return [doctor.Finding("config", doctor.PASS, "parsed")]

    monkeypatch.setattr(doctor, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["doctor", "--config", "custom.yaml", "--no-listen", "--json"])
    assert doctor.main() == 0
    assert json.loads(capsys.readouterr().out)[0]["status"] == doctor.PASS


@pytest.mark.parametrize("module", ["yam_abc_reproduce.doctor", "scripts.setup_can_udev"])
def test_selftest_exits_without_hardware(module, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [module, "--selftest"])
    assert importlib.import_module(module).main() == 0
    assert "selftest OK" in capsys.readouterr().out


def test_bad_home_pose_fails_before_building_arms(monkeypatch, capsys):
    from yam_abc_reproduce import config, runtime
    from yam_abc_reproduce.deploy.run import deploy

    monkeypatch.setattr(config, "build_station_config", lambda *args: config.StationConfig())
    monkeypatch.setattr(
        runtime, "build_arm_units", lambda *args, **kwargs: pytest.fail("opened hardware")
    )
    with pytest.raises(SystemExit) as exc:
        deploy(["--host", "localhost", "--port", "8000", "--prompt", "pick", "--home-pose", "0"])
    assert exc.value.code == 2
    assert "--home-pose has 1 values" in capsys.readouterr().err
