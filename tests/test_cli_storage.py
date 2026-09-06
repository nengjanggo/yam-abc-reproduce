"""Standalone artifact commands share the GUI's data/model storage layout."""

import builtins
import importlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from yam_abc_reproduce import cli
from yam_abc_reproduce.config import StationConfig
from yam_abc_reproduce.storage import REPO_ROOT, StoragePaths


@pytest.fixture(autouse=True)
def isolate_cache_environment(monkeypatch):
    for key in StoragePaths().cache_env():
        monkeypatch.setenv(key, f"/previous/{key}")
    monkeypatch.setenv("HF_HOME", "/existing/auth")


@pytest.fixture
def paths(tmp_path):
    return StoragePaths(tmp_path / "save root 'quoted'")


def _load_script(name, monkeypatch):
    spec = importlib.util.spec_from_file_location(f"storage_test_{name}", REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("custom", [False, True])
@pytest.mark.parametrize("fmt", ["lerobot", "abc"])
def test_convert_defaults_and_cache_environment(paths, tmp_path, monkeypatch, custom, fmt):
    from yam_abc_reproduce.data import formats

    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr(formats, "convert_episode", lambda *a, **kw: calls.append((a, kw)))
    args = ["input/episodes", "--to", fmt, "--repo-id", "org/fold"]
    if custom:
        args += ["--save-root", str(paths.root)]
    cli.convert(args)
    expected = paths if custom else StoragePaths()
    assert calls == [((Path("input/episodes"),), {
        "to": fmt, "repo_id": "org/fold", "out": str(expected.data / fmt / "org/fold")})]
    for key, value in expected.cache_env().items():
        assert os.environ[key] == value
    assert os.environ["HF_HOME"] == "/existing/auth"


@pytest.mark.parametrize("out", ["exports/dataset", "/external/dataset", "~/exported-dataset"])
def test_convert_explicit_output(paths, monkeypatch, out):
    from yam_abc_reproduce.data import formats

    calls = []
    monkeypatch.setattr(formats, "convert_episode", lambda *a, **kw: calls.append(kw))
    cli.convert(["input", "--save-root", str(paths.root), "--out", out])
    assert calls[0]["out"] == str(paths.resolve(out))


@pytest.mark.parametrize("configured", ["data/episodes", "custom/episodes", "/external/episodes"])
def test_teleop_recording_uses_resolved_station_path(paths, monkeypatch, configured):
    from yam_abc_reproduce import config, runtime
    from yam_abc_reproduce.teleop import loop

    cfg = StationConfig(save_root=configured)
    monkeypatch.setattr(config, "build_station_config", lambda *a: cfg)
    monkeypatch.setattr(runtime, "build_arm_units", lambda *a, **kw: [])
    monkeypatch.setattr(runtime, "build_cameras_from_config", lambda *a, **kw: [])
    calls = []
    monkeypatch.setattr(loop, "ControlLoop", lambda *a, **kw: SimpleNamespace(
        run_for=lambda **kw: calls.append(kw)))
    cli.teleop(["--save-root", str(paths.root), "--record", "fold", "--mock"])
    assert calls[0]["save_root"] == str(paths.resolve(configured))
    assert calls[0]["station"].save_root == calls[0]["save_root"]
    assert calls[0]["record_task"] == "fold"


@pytest.mark.parametrize("root", [None, "explicit/dataset"])
def test_viz_resolves_dataset_without_rebasing_explicit_input(paths, monkeypatch, root):
    from yam_abc_reproduce.data import visualize

    calls = []
    monkeypatch.setattr(visualize, "visualize_lerobot", lambda **kw: calls.append(kw))
    cli.viz(["--save-root", str(paths.root), "--repo-id", "org/fold",
             *(["--root", root] if root else [])])
    assert calls[0]["root"] == (root or str(paths.lerobot / "org/fold"))


def test_norm_stats_default_cache_and_output_override(paths, monkeypatch):
    module = _load_script("compute_abc_norm_stats", monkeypatch)
    episode = paths.abc_cache / "train_real/fold_ep"
    episode.mkdir(parents=True)
    (episode / "episode_metadata.json").write_text('{"num_steps": 3}')
    np.arange(84, dtype=np.float64).reshape(3, 28).tofile(episode / "states_actions.bin")
    module.main(["--save-root", str(paths.root)])
    stats = json.loads((paths.abc_cache / "norm_stats.json").read_text())
    assert len(stats["state"]["mean"]) == len(stats["actions"]["mean"]) == 14
    module.main(["--save-root", str(paths.root), "--out", "data/custom/norm.json"])
    assert json.loads((paths.data / "custom/norm.json").read_text()) == stats


def test_raw_abc_conversion_layout(paths, monkeypatch):
    module = _load_script("yam_episodes_to_abc_mcap", monkeypatch)
    monkeypatch.setattr(module, "_find_episodes", lambda src: [src / "ep1", src / "ep2"])
    monkeypatch.setattr(module, "_load_classes", dict)
    outputs = []
    monkeypatch.setattr(module, "convert_episode", lambda src, dst, *args: outputs.append(dst))
    module.main(["--save-root", str(paths.root), "--src", "/raw", "--task", "fold", "--val", "1"])
    assert outputs == [paths.data / "abc_release/val/fold/episode_000000/episode.mcap",
                       paths.data / "abc_release/train/fold/episode_000001/episode.mcap"]


@pytest.mark.parametrize("dst", [None, "model/custom.pt"])
def test_merge_checkpoint_default_and_override(paths, monkeypatch, dst):
    from yam_abc_reproduce.deploy import merge_abc_lora

    calls = []
    monkeypatch.setattr(merge_abc_lora, "merge_checkpoint", lambda *a: calls.append(a))
    merge_abc_lora.main(["input/run.pt", *([dst] if dst else []), "--save-root", str(paths.root)])
    assert calls == [(Path("input/run.pt"), paths.resolve(dst) if dst else paths.model / "abc/merged/run_merged.pt")]


@pytest.mark.parametrize("name,filename", [
    ("check_can_usb_quality", "can_usb_quality.json"),
    ("monitor_can_reply_latency", "can_reply_latency.json"),
])
def test_diagnostic_report_paths(paths, monkeypatch, name, filename):
    module = _load_script(name, monkeypatch)
    ticks = iter(range(100))
    monkeypatch.setattr(module.time, "monotonic", lambda: next(ticks))
    if name == "check_can_usb_quality":
        monkeypatch.setattr(module, "link_snapshot", lambda iface: {"interface": iface})
    else:
        class FakeSocket:
            def close(self):
                pass

        monkeypatch.setattr(module, "open_can_socket", lambda iface: FakeSocket())
    module.main(["--save-root", str(paths.root), "--seconds", "0.01", "--interfaces", "fake"])
    assert json.loads((paths.data / "diagnostics" / filename).read_text())
    module.main(["--save-root", str(paths.root), "--out", "data/custom/report.json",
                 "--seconds", "0.01", "--interfaces", "fake"])
    assert (paths.data / "custom/report.json").is_file()


@pytest.mark.parametrize("module_name,dependency,args", [
    ("openpi_server", "openpi.policies", ["--config", "pi0_yam", "--checkpoint", "/weights"]),
    ("molmoact_server", "torch", []),
    ("abc_server", "abc_minimal.config", ["--checkpoint", "/weights", "--prompt", "fold"]),
])
def test_servers_configure_caches_before_backend_import(paths, monkeypatch, module_name, dependency, args):
    module = importlib.import_module(f"yam_abc_reproduce.deploy.servers.{module_name}")
    original = builtins.__import__

    class BackendReached(Exception):
        pass

    def intercept(name, *args, **kwargs):
        if name == dependency:
            for key, value in paths.cache_env().items():
                assert os.environ[key] == value
            assert os.environ["HF_HOME"] == "/existing/auth"
            raise BackendReached
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", intercept)
    with pytest.raises(BackendReached):
        module.main(["--save-root", str(paths.root), *args])


@pytest.mark.parametrize("cache_override", [None, "model/custom-abc"])
def test_abc_server_model_cache_default_and_override(paths, monkeypatch, cache_override):
    from yam_abc_reproduce.deploy.servers import abc_server

    cfg = SimpleNamespace(checkpoint="/weights", prompt="fold", clip=SimpleNamespace(), model=SimpleNamespace(
        camera_keys=["top"], state_dim=14, action_dim=14, chunk_length=30))
    monkeypatch.setitem(sys.modules, "abc_minimal.config", SimpleNamespace(SimEvalConfig=lambda **kw: cfg))
    calls = []

    def local_checkpoint(path, cache):
        calls.append(cache)
        return Path(path)

    monkeypatch.setitem(sys.modules, "abc_minimal.eval_policy", SimpleNamespace(
        SimPolicy=lambda *a, **kw: None, local_checkpoint=local_checkpoint))
    monkeypatch.setitem(sys.modules, "abc_minimal.dit", SimpleNamespace(infer_dit_shape=lambda path: {}))
    monkeypatch.setattr(abc_server, "_maybe_merge_lora", lambda path, cache: calls.append(cache) or path)
    monkeypatch.setattr(abc_server._wire, "serve", lambda *a: None)
    abc_server.main(["--save-root", str(paths.root), "--checkpoint", "/weights", "--prompt", "fold",
                     *(["--model-cache-root", cache_override] if cache_override else [])])
    cache = paths.resolve(cache_override) if cache_override else paths.model_cache("abc")
    assert calls == [cache / "downloads", cache / "merged"]
    assert cfg.clip.cache_dir == str(cache / "clip")


@pytest.mark.parametrize("relative", [
    "scripts/compute_abc_norm_stats.py", "scripts/yam_episodes_to_abc_mcap.py",
    "scripts/check_can_usb_quality.py", "scripts/monitor_can_reply_latency.py",
    "yam_abc_reproduce/deploy/merge_abc_lora.py",
    "yam_abc_reproduce/deploy/servers/openpi_server.py",
    "yam_abc_reproduce/deploy/servers/molmoact_server.py",
    "yam_abc_reproduce/deploy/servers/abc_server.py",
])
def test_standalone_help_from_another_directory(tmp_path, relative):
    result = subprocess.run([sys.executable, str(REPO_ROOT / relative), "--help"], cwd=tmp_path,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert "--save-root ROOT" in result.stdout
