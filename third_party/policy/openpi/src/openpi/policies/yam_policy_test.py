from typing import Any

import numpy as np

from openpi.models import model as _model
from openpi.policies import yam_policy


def test_single_arm_missing_right_wrist_is_masked(
) -> None:
    '''Single-arm sample을 변환하고 비어 있는 right wrist slot을 mask하는지 검증한다.'''
    image_height: int = 224
    image_width: int = 224
    image_channels: int = 3
    state_dim: int = 7
    action_horizon: int = 50

    # Single-arm LeRobot sample과 동일한 shape의 입력을 구성한다.
    top_image: np.ndarray = np.full(
        (image_height, image_width, image_channels),
        17,
        dtype=np.uint8,
    )  # shape: (image_height, image_width, image_channels)
    wrist_image: np.ndarray = np.full(
        (image_height, image_width, image_channels),
        23,
        dtype=np.uint8,
    )  # shape: (image_height, image_width, image_channels)
    state: np.ndarray = np.zeros((state_dim,), dtype=np.float32)  # shape: (state_dim,)
    actions: np.ndarray = np.zeros(
        (action_horizon, state_dim),
        dtype=np.float32,
    )  # shape: (action_horizon, state_dim)
    data: dict[str, Any] = {
        'observation/image': top_image,
        'observation/left_wrist': wrist_image,
        'observation/state': state,
        'actions': actions,
        'prompt': 'pick up the white ethernet cable',
    }

    # 기존 YAM policy transform을 실행한다.
    transformed: dict[str, Any] = yam_policy.YamInputs(
        model_type=_model.ModelType.PI05,
    )(data)

    # 사용하지 않는 right wrist slot의 값과 mask contract를 검증한다.
    np.testing.assert_array_equal(transformed['image']['base_0_rgb'], top_image)
    np.testing.assert_array_equal(transformed['image']['left_wrist_0_rgb'], wrist_image)
    np.testing.assert_array_equal(transformed['image']['right_wrist_0_rgb'], np.zeros_like(top_image))
    assert transformed['image_mask'] == {
        'base_0_rgb': np.True_,
        'left_wrist_0_rgb': np.True_,
        'right_wrist_0_rgb': np.False_,
    }
    np.testing.assert_array_equal(transformed['state'], state)
    np.testing.assert_array_equal(transformed['actions'], actions)
