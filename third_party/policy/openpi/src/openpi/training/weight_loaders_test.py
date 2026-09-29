from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import model as _model
import openpi.shared.array_typing as at
from openpi.training import weight_loaders


def test_checkpoint_weight_loader_restores_target_dtype_without_conversion_copy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    '''Checkpoint parameter를 target dtype으로 직접 복원하고 기존 cast 결과와 일치하는지 검증한다.'''
    # FP32 checkpoint parameter와 BF16 target parameter shape을 구성한다.
    source_base: np.ndarray = np.array([0.1, -0.2, 1.5, 3.25], dtype=np.float32)
    source_params: at.Params = {
        'base': source_base,
    }
    reference_params: at.Params = {
        'base': jax.ShapeDtypeStruct((4,), jnp.bfloat16),
        'action_expert_lora': np.array([0.25], dtype=np.float32),
    }
    checkpoint_dir: Path = tmp_path / 'params'
    checkpoint_dir.mkdir()
    restored_dtypes: list[jnp.dtype | None] = []
    restored_types: list[type[np.ndarray] | type[jax.Array]] = []

    def fake_restore_params(
        params_path: Path | str,
        *,
        restore_type: type[np.ndarray] | type[jax.Array],
        dtype: jnp.dtype | None = None,
    ) -> at.Params:
        '''Checkpoint 경로와 dtype을 받아 Orbax restore 결과와 같은 parameter tree를 반환한다.'''
        assert Path(params_path) == checkpoint_dir
        restored_types.append(restore_type)
        restored_dtypes.append(dtype)
        return {
            key: (
                jnp.asarray(value, dtype=dtype)
                if restore_type is jax.Array
                else value.astype(dtype) if dtype is not None else value.copy()
            )
            for key, value in source_params.items()
        }

    monkeypatch.setattr(_model, 'restore_params', fake_restore_params)

    # 기존 restore 후 cast와 같은 BF16 기준값을 계산한다.
    baseline_base: np.ndarray = source_base.astype(jnp.bfloat16)
    loader: weight_loaders.CheckpointWeightLoader = weight_loaders.CheckpointWeightLoader(
        str(checkpoint_dir),
        restore_dtype='bfloat16',
        restore_as_jax_array=True,
    )
    restored_params: at.Params = loader.load(reference_params)

    # Direct restore dtype과 최종 parameter 값이 기존 경로와 정확히 같은지 검증한다.
    assert restored_dtypes == [jnp.dtype(jnp.bfloat16)]
    assert restored_types == [jax.Array]
    assert restored_params['base'].dtype == jnp.bfloat16
    np.testing.assert_array_equal(restored_params['base'], baseline_base)
    np.testing.assert_array_equal(restored_params['action_expert_lora'], reference_params['action_expert_lora'])
