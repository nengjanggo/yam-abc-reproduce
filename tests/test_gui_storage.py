"""Storage routing without robot hardware, model downloads, or GPU training."""

import ast
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from yam_abc_reproduce.config import CameraConfig, RobotConfig, StationConfig  # noqa: E402
from yam_abc_reproduce.gui import builders, convert_progress  # noqa: E402
from yam_abc_reproduce.gui.jobs import JobManager  # noqa: E402
from yam_abc_reproduce.gui.server import create_app  # noqa: E402
from yam_abc_reproduce.gui.storage import REPO_ROOT, StoragePaths  # noqa: E402


def test_root_resolution(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert StoragePaths().root == REPO_ROOT
    assert StoragePaths(Path("relative root")).root == tmp_path / "relative root"
    assert StoragePaths(Path("~/yam-storage")).root == Path.home() / "yam-storage"
    paths = StoragePaths(tmp_path)
    assert paths.resolve("custom/data") == tmp_path / "custom/data"
    assert paths.resolve("/external/data") == Path("/external/data")


@pytest.mark.parametrize("custom", [False, True])
def test_gui_cli_passes_save_root(tmp_path, monkeypatch, custom):
    import uvicorn

    from yam_abc_reproduce.cli import gui

    apps = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: apps.append(app))
    gui(["--mock", *(["--save-root", str(tmp_path)] if custom else [])])
    assert apps[0].state.storage.root == (tmp_path if custom else REPO_ROOT)
    assert apps[0].state.session.cfg.save_root == str(apps[0].state.storage.episodes)


def _station():
    return StationConfig(
        robot=RobotConfig(type="mock", num_arm_joints=6),
        cameras=[CameraConfig(name="top", type="mock", role="top", mode="mono")],
        control_hz=100,
    )


def test_recording_and_review_under_root(tmp_path):
    app = create_app(_station(), mock=True, save_root=tmp_path)
    with TestClient(app) as client:
        assert client.post("/api/collect/start-teleop").status_code == 200
        time.sleep(0.15)
        res = client.post("/api/collect/start-recording", json={"task_name": "fold"}).json()
        episode = Path(res["episode"])
        assert episode.parent == tmp_path / "data/episodes/fold"
        time.sleep(0.15)
        assert client.post("/api/collect/stop-recording").json()["frames"] > 0
        assert (episode / "write_complete.flag").exists()
        rollout = tmp_path / "data/rollouts/abc/fold/episode_1"
        rollout.mkdir(parents=True)
        (rollout / "write_complete.flag").touch()
        rows = client.get("/api/review/episodes").json()["episodes"]
        assert {row["source"] for row in rows} == {"dataset", "rollout"}


def test_reload_overrides_and_app_isolation(tmp_path, monkeypatch):
    from yam_abc_reproduce.gui import server

    original = _station()
    monkeypatch.setattr(server, "build_station_config", lambda *args: original)
    apps = [create_app(original, mock=True, save_root=tmp_path / name,
                       station_path="unused.yaml") for name in ("one", "two")]
    for app in apps:
        paths = app.state.storage
        # Observe the config passed to the camera session without opening devices.
        seen = []
        monkeypatch.setattr(app.state.session, "connect_cameras", lambda cfg: seen.append(cfg))
        monkeypatch.setattr(app.state.session, "start_deploy", lambda **kw: seen.append(kw))
        with TestClient(app) as client:
            cfg = client.get("/api/config").json()
            assert cfg["storage"]["root"] == str(paths.root)
            assert cfg["save_root"] == str(paths.episodes)
            for body in (None, {}):
                assert client.post("/api/collect/connect", json=body).status_code == 200
                assert seen[-1].save_root == str(paths.episodes)
            assert client.post("/api/collect/connect", json={"save_root": "custom"}).status_code == 200
            assert seen[-1].save_root == str(paths.root / "custom")
            assert client.post("/api/collect/connect", json={"save_root": "/external/data"}).status_code == 200
            assert seen[-1].save_root == "/external/data"
            for body, expected in (({}, paths.rollouts / "pi0"),
                                   ({"save_root": "data/rollouts/abc"}, paths.rollouts / "abc"),
                                   ({"save_root": "/external/rollouts"}, Path("/external/rollouts"))):
                assert client.post("/api/deploy/start", json=body).status_code == 200
                assert seen[-1]["save_root"] == str(expected)
            note = client.get("/api/train/fields?backend=abc").json()["note"]
            assert str(paths.abc_cache) in note
    assert original.save_root == "data/episodes"  # factory never mutates caller config


@pytest.fixture
def command_capture(tmp_path, monkeypatch):
    """Execute real shell commands, substituting only the backend executables."""
    venv = tmp_path / "fake venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "lib/python3/site-packages/nvidia/mock/lib").mkdir(parents=True)
    keys = [*StoragePaths().cache_env(), "MOLMO_DATA_DIR", "ABC_DEBUG_DUMP", "HF_HOME"]
    program = (
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        f"env = {{key: os.environ.get(key) for key in {keys!r}}}\n"
        "print(json.dumps({'argv': sys.argv[1:], 'env': env}))\n"
    )
    for name in ("python", "torchrun", "yam-abc-convert"):
        path = venv / "bin" / name
        path.write_text(program)
        path.chmod(0o755)
    monkeypatch.setattr(builders, "_VENV", venv)
    monkeypatch.setattr(builders, "_PY", venv / "bin/python")
    monkeypatch.setattr(builders, "_TORCHRUN", venv / "bin/torchrun")
    monkeypatch.setattr(builders, "_require_backend_venv", lambda backend: None)
    monkeypatch.setattr(builders._gpus, "pick", lambda n: ",".join(str(i) for i in range(n)))
    monkeypatch.setenv("HF_HUB_CACHE", "/previous/model/cache")
    monkeypatch.setenv("HF_HOME", "/existing/auth/location")

    def execute(cmd):
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=True)
        return [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]

    return execute


@pytest.mark.parametrize("backend", ["pi0", "pi05", "molmoact2", "abc"])
def test_backend_shell_paths_and_cache_environment(tmp_path, command_capture, backend):
    paths = StoragePaths(tmp_path / "save 'root $literal")
    (paths.lerobot / "org/fold").mkdir(parents=True)
    for name in ("train_real", "val_real"):
        (paths.abc_cache / name).mkdir(parents=True)
    (paths.abc_cache / "norm_stats.json").write_text("{}")
    params = {"backend": backend, "dataset": "org/fold", "run_name": "trial 'one",
              "checkpoint": "/input/checkpoint with spaces", "prompt": "fold cloth"}
    train = command_capture(builders.build_train_command(params, paths)[0])
    deploy = command_capture(builders.build_deploy_command(params, paths)[0])
    for child in [*train, *deploy]:
        for key, expected in paths.cache_env().items():
            assert child["env"][key] == expected
        assert child["env"]["HF_HOME"] == "/existing/auth/location"
    assert os.environ["HF_HUB_CACHE"] == "/previous/model/cache"
    argv = train[-1]["argv"]
    if backend in ("pi0", "pi05"):
        assert f"--assets-base-dir={paths.openpi_assets}" in train[0]["argv"]
        assert f"--assets-base-dir={paths.openpi_assets}" in argv
        assert f"--checkpoint-base-dir={paths.checkpoints(backend)}" in argv
    elif backend == "molmoact2":
        assert f"--save_folder={paths.checkpoints(backend) / params['run_name']}" in argv
        assert train[-1]["env"]["MOLMO_DATA_DIR"] == str(paths.data / "molmo_data")
    else:
        assert f"--cache-root={paths.abc_cache}" in argv
        assert f"--checkpoint-dir={paths.checkpoints(backend) / params['run_name']}" in argv
        assert f"--clip.cache-dir={paths.model_cache('abc') / 'clip'}" in argv
        assert f"--model-cache-root={paths.model_cache('abc')}" in argv
        assert deploy[0]["env"]["ABC_DEBUG_DUMP"] == str(paths.data / "abc_debug")
        assert str(paths.model_cache("abc")) in deploy[0]["argv"]


@pytest.mark.parametrize("fmt", ["lerobot", "abc"])
def test_conversion_job_freezes_paths_for_progress(tmp_path, command_capture, fmt):
    paths = StoragePaths(tmp_path / "save root")
    # Explicit relative overrides are rooted consistently, including progress.
    params = paths.conversion_params({"src": "custom/task", "to": fmt, "out": "exports/ds"})
    episode = Path(params["src"]) / "ep"
    episode.mkdir(parents=True)
    (episode / "write_complete.flag").touch()
    (episode / "metadata.json").write_text('{"num_frames": 15}')
    manager = JobManager(paths)
    job = manager.launch("convert", params)
    job._thread.join(timeout=10)
    assert job.returncode == 0
    child = json.loads(job.logs[0])
    assert child["argv"][0] == str(paths.root / "custom/task")
    assert child["argv"][-2:] == ["--out", str(paths.root / "exports/ds")]
    out = Path(params["out"])
    out.mkdir(parents=True)
    if fmt == "abc":
        (out / "episode_000000.mcap").touch()
    else:
        (out / "meta").mkdir()
        (out / "meta/info.json").write_text('{"total_episodes": 1, "total_frames": 15}')
    snap = convert_progress.snapshot(job.params, None, job.started, paths)
    assert snap["episodes_done"] == snap["episodes_total"] == 1
    assert snap["frac"] == 1.0


def test_gui_conversion_uses_current_collection_override(tmp_path, monkeypatch):
    app = create_app(StationConfig(save_root="custom recordings"), mock=True, save_root=tmp_path)
    seen = []

    class FakeJob:
        def summary(self):
            return {"id": "test"}

    def launch(kind, params):
        seen.append(params)
        return FakeJob()

    monkeypatch.setattr(app.state.jobs, "launch", launch)
    with TestClient(app) as client:
        for fmt in ("lerobot", "abc"):
            res = client.post("/api/jobs", json={"kind": "convert", "params": {
                "task": "fold", "repo_id": "org/fold", "to": fmt}})
            assert res.status_code == 200
            assert seen[-1]["src"] == str(tmp_path / "custom recordings/fold")
            assert seen[-1]["out"] == str(tmp_path / "data" / fmt / "org/fold")


def test_checkpoint_discovery_is_per_app_and_absolute(tmp_path):
    roots = [tmp_path / name for name in ("one", "two")]
    for root in roots:
        paths = StoragePaths(root)
        expected = {
            "pi0": paths.checkpoints("pi0") / "pi0_yam/run/100",
            "pi05": paths.checkpoints("pi05") / "pi05_yam/run/200",
            "molmoact2": paths.checkpoints("molmoact2") / "run/step300",
            "abc": paths.checkpoints("abc") / "run/400.pt",
        }
        for backend, path in expected.items():
            if backend in ("pi0", "pi05"):
                (path / "params").mkdir(parents=True)
            elif backend == "abc":
                path.parent.mkdir(parents=True)
                path.touch()
            else:
                path.mkdir(parents=True)
        with TestClient(create_app(_station(), mock=True, save_root=root)) as client:
            for backend, path in expected.items():
                result = client.get(f"/api/deploy/checkpoints?backend={backend}").json()
                assert result["checkpoints"] == [str(path)]


def test_abc_config_options_preserve_standalone_defaults(monkeypatch):
    path = REPO_ROOT / "third_party/policy/abc/abc_minimal/config.py"
    spec = importlib.util.spec_from_file_location("abc_storage_config", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    config = module.TrainConfig()
    assert config.model_cache_root is None and config.checkpoint_dir is None
    tyro = pytest.importorskip("tyro")
    config = tyro.cli(module.TrainConfig, args=[
        "--model-cache-root=/tmp/model assets", "--checkpoint-dir=/tmp/run checkpoints",
        "--clip.cache-dir=/tmp/clip assets"])
    assert config.model_cache_root == "/tmp/model assets"
    assert config.checkpoint_dir == "/tmp/run checkpoints"
    assert config.clip.cache_dir == "/tmp/clip assets"


def test_openpi_norm_stats_passes_asset_override_to_data_factory(tmp_path):
    """Exercise the actual entrypoint with lightweight stand-ins for GPU imports."""
    import dataclasses
    from types import SimpleNamespace

    source = REPO_ROOT / "third_party/policy/openpi/scripts/compute_norm_stats.py"
    main = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "main")
    seen = []

    @dataclasses.dataclass
    class Config:
        assets_base_dir: str = "./assets"
        model = SimpleNamespace(action_horizon=50)
        batch_size = 1
        num_workers = 0
        data = SimpleNamespace(create=lambda assets, model: (
            seen.append(assets) or SimpleNamespace(rlds_data_dir=None, asset_id="yam")))

        @property
        def assets_dirs(self):
            return Path(self.assets_base_dir) / "pi0_yam"

    env = {
        "dataclasses": dataclasses,
        "_config": SimpleNamespace(get_config=lambda name: Config()),
        "create_torch_dataloader": lambda *args: ([], 0),
        "normalize": SimpleNamespace(
            RunningStats=lambda: SimpleNamespace(get_statistics=lambda: {}),
            save=lambda path, stats: seen.append(path)),
        "tqdm": SimpleNamespace(tqdm=lambda data, **kw: data),
    }
    exec(compile(ast.Module(body=[main], type_ignores=[]), str(source), "exec"), env)
    env["main"]("pi0_yam", assets_base_dir=str(tmp_path))
    assert seen == [tmp_path / "pi0_yam", tmp_path / "pi0_yam/yam"]


def test_abc_deploy_downloads_and_merges_under_model_cache(tmp_path, monkeypatch):
    """Exercise backend file writes with fake downloads/tensors, keeping external inputs intact."""
    import hashlib
    import logging
    from types import SimpleNamespace

    import numpy as np

    def load_function(source, name, env):
        node = next(n for n in ast.parse(source.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), env)
        return env[name]

    paths = StoragePaths(tmp_path / "save root")
    cache = paths.model_cache("abc")
    downloads = []

    def download(cmd, **kwargs):
        downloads.append(cmd)
        Path(cmd[-1]).write_bytes(b"checkpoint")

    local_checkpoint = load_function(
        REPO_ROOT / "third_party/policy/abc/abc_minimal/eval_policy.py", "local_checkpoint",
        {"Path": Path, "ROOT": tmp_path / "legacy", "subprocess": SimpleNamespace(run=download)})
    checkpoint = local_checkpoint("s3://weights/run.pt", cache / "downloads")
    assert checkpoint == cache / "downloads/run.pt" and checkpoint.is_file()
    assert local_checkpoint("s3://weights/run.pt", cache / "downloads") == checkpoint
    assert len(downloads) == 1

    class Tensor(np.ndarray):
        def float(self):
            return self.astype(np.float32)

        def to(self, dtype):
            return self.astype(dtype)

    weights = {"model": {"layer.base.weight": np.ones((2, 2)).view(Tensor),
                         "layer.lora_a.weight": np.ones((1, 2)).view(Tensor),
                         "layer.lora_b.weight": np.ones((2, 1)).view(Tensor)}, "norm_stats": {"x": 1}}
    saved = []

    def save(value, path):
        saved.append(value)
        Path(path).write_bytes(b"merged")

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(load=lambda *a, **kw: weights, save=save))
    merge = load_function(
        REPO_ROOT / "yam_abc_reproduce/deploy/servers/abc_server.py", "_maybe_merge_lora",
        {"Path": Path, "os": os, "hashlib": hashlib, "log": logging.getLogger(__name__)})
    external = tmp_path / "external.pt"
    external.write_bytes(b"original")
    merged = Path(merge(external, cache / "merged"))
    assert merged.parent == cache / "merged" and merged.is_file()
    assert external.read_bytes() == b"original"
    assert not external.with_name("external_merged.pt").exists()
    assert saved[0]["norm_stats"] == {"x": 1}
    assert np.array_equal(saved[0]["model"]["layer.weight"], np.full((2, 2), 2))
    assert merge(external, cache / "merged") == str(merged)
    assert len(saved) == 1
    external.write_bytes(b"updated checkpoint")
    assert Path(merge(external, cache / "merged")) != merged
