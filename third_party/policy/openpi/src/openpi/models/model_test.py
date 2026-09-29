from flax import nnx
import jax
import numpy as np
import pytest

from openpi.models import model as _model
from openpi.models import pi0
from openpi.models import pi0_config
from openpi.models import pi0_fast
from openpi.shared import download
from openpi.shared import nnx_utils


def test_pi0_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_lora_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi05_stop_gradient_vlm_prefix_preserves_loss_and_action_expert_gradients(
) -> None:
    '''같은 model과 입력에서 stop_gradient 적용 전후 loss와 action expert gradient를 검증한다.'''
    def mean_loss(
        model: pi0.Pi0,
        rng: jax.Array,
        observation: _model.Observation,
        actions: _model.Actions,
    ) -> jax.Array:
        '''Model과 batch를 받아 action horizon 전체의 mean loss를 반환한다.'''
        loss: jax.Array = model.compute_loss(rng, observation, actions)
        return loss.mean()

    rng: jax.Array = jax.random.key(0)
    config: pi0_config.Pi0Config = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant='dummy',
        action_expert_variant='dummy',
    )
    model: pi0.Pi0 = config.create(rng)
    batch_size: int = 1
    observation: _model.Observation
    actions: _model.Actions
    observation, actions = config.fake_obs(batch_size), config.fake_act(batch_size)
    trainable_filter: nnx.filterlib.Filter = nnx.All(
        nnx.Param,
        nnx_utils.PathRegex('.*llm.*_1.*'),
    )
    diff_state: nnx.DiffState = nnx.DiffState(0, trainable_filter)

    # 같은 parameter, RNG, batch로 baseline loss와 action expert gradient를 계산한다.
    model.stop_gradient_vlm_prefix = False
    baseline_loss: jax.Array
    baseline_gradients: nnx.State
    baseline_loss, baseline_gradients = nnx.value_and_grad(
        mean_loss,
        argnums=diff_state,
    )(model, rng, observation, actions)

    # Frozen VLM autodiff 연결을 차단한 loss와 action expert gradient를 계산한다.
    model.stop_gradient_vlm_prefix = True
    optimized_loss: jax.Array
    optimized_gradients: nnx.State
    optimized_loss, optimized_gradients = nnx.value_and_grad(
        mean_loss,
        argnums=diff_state,
    )(model, rng, observation, actions)

    np.testing.assert_array_equal(np.asarray(optimized_loss), np.asarray(baseline_loss))
    baseline_gradient_leaves: list[jax.Array] = jax.tree.leaves(baseline_gradients)
    optimized_gradient_leaves: list[jax.Array] = jax.tree.leaves(optimized_gradients)
    assert baseline_gradient_leaves
    assert len(optimized_gradient_leaves) == len(baseline_gradient_leaves)
    for baseline_gradient, optimized_gradient in zip(
        baseline_gradient_leaves,
        optimized_gradient_leaves,
        strict=True,
    ):
        np.testing.assert_array_equal(
            np.asarray(optimized_gradient),
            np.asarray(baseline_gradient),
        )


def test_pi0_fast_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)


def test_pi0_fast_lora_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)

    lora_filter = nnx_utils.PathRegex(".*lora.*")
    model_state = nnx.state(model)

    lora_state_elems = list(model_state.filter(lora_filter))
    assert len(lora_state_elems) > 0


@pytest.mark.manual
def test_model_restore():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    model = config.load(
        _model.restore_params(download.maybe_download("gs://openpi-assets/checkpoints/pi0_base/params"))
    )

    loss = model.compute_loss(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = model.sample_actions(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)
