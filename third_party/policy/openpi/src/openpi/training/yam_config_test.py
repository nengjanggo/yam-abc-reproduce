from typing import Any

import flax.nnx as nnx
import jax
import pytest

from openpi.models import gemma
from openpi.models import lora as _lora
from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import weight_loaders


def test_pi05_yam_dit_lora_config_selects_only_action_expert_lora(
) -> None:
    '''YAM config를 받아 action expert LoRA parameter만 trainable인지 검증한다.'''
    # 등록된 fine-tuning config와 abstract model state를 읽는다.
    config: _config.TrainConfig = _config.get_config('pi05_yam_dit_lora')
    assert isinstance(config.model, pi0_config.Pi0Config)
    abstract_model: Any = nnx.eval_shape(config.model.create, jax.random.key(0))
    trainable_paths: list[str] = [
        '/'.join(str(path_part) for path_part in path)
        for path in nnx.state(abstract_model, config.trainable_filter).flat_state()
    ]

    # VLM parameter를 제외하고 action expert LoRA parameter만 남는지 검증한다.
    assert trainable_paths
    assert all('llm' in path for path in trainable_paths)
    assert all('_1' in path for path in trainable_paths)
    assert all('lora' in path for path in trainable_paths)
    assert config.model.paligemma_variant == 'gemma_2b'
    assert config.model.action_expert_variant == 'gemma_300m_lora'
    assert config.model.action_expert_lora_rank == 32
    assert config.model.action_expert_lora_alpha == 32.0
    assert not config.model.action_expert_lora_rslora
    assert config.model.stop_gradient_vlm_prefix
    assert isinstance(config.data, _config.LeRobotYamDataConfig)
    assert config.data.repo_id == (
        'nengjanggo/yam_pick_up_the_white_ethernet_cable_and_plug_it_into_the_black_ethernet_port'
    )
    assert config.data.num_arms == 1
    assert isinstance(config.weight_loader, weight_loaders.CheckpointWeightLoader)
    assert config.weight_loader.restore_dtype == 'bfloat16'
    assert config.weight_loader.restore_as_jax_array


def test_gemma_lora_hyperparameter_override(
) -> None:
    '''LoRA rank, alpha와 rsLoRA override가 attention과 FFN에 적용되는지 검증한다.'''
    lora_rank: int = 7
    lora_alpha: float = 14.0

    # Non-default LoRA hyperparameter로 action expert config를 생성한다.
    action_expert_config: gemma.Config = gemma.get_config(
        'gemma_300m_lora',
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_rslora=True,
    )

    # Attention과 FFN이 같은 LoRA hyperparameter를 공유하는지 검증한다.
    lora_configs: list[_lora.LoRAConfig] = list(action_expert_config.lora_configs.values())
    assert len(lora_configs) == 2
    assert all(lora_config.rank == lora_rank for lora_config in lora_configs)
    assert all(lora_config.alpha == lora_alpha for lora_config in lora_configs)
    assert all(lora_config.rslora for lora_config in lora_configs)


@pytest.mark.parametrize(
    ('lora_rank', 'lora_alpha', 'error_pattern'),
    [
        (0, 32.0, 'lora_rank'),
        (32, 0.0, 'lora_alpha'),
    ],
)
def test_gemma_lora_hyperparameter_validation(
    lora_rank: int,
    lora_alpha: float,
    error_pattern: str,
) -> None:
    '''유효하지 않은 LoRA rank와 alpha가 model 생성 전에 거부되는지 검증한다.'''
    with pytest.raises(ValueError, match=error_pattern):
        gemma.get_config(
            'gemma_300m_lora',
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )
