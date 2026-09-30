# Real-Time Chunking (RTC) soft-masking guidance.
#
# Source: https://github.com/Physical-Intelligence/real-time-chunking-kinetix
#   commit 9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b, file src/model.py
#   - `PrefixAttentionSchedule`, `get_prefix_weights`: copied verbatim.
#   - `pinv_corrected_velocity`: copied from `FlowPolicy.realtime_action`. Changes for openpi:
#     the model call is passed in as `velocity_fn` instead of `self(obs, ...)`, and the per-example
#     `jax.vmap` is removed because the openpi denoiser is already batched and independent per example.
#     Time convention is unchanged (t=0 noise, t=1 data); callers in openpi convert from their own.
#
# MIT License
#
# Copyright (c) 2025 Physical Intelligence
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from collections.abc import Callable
from typing import Literal, TypeAlias

import jax
import jax.numpy as jnp

PrefixAttentionSchedule: TypeAlias = Literal["linear", "exp", "ones", "zeros"]


def get_prefix_weights(start: int, end: int, total: int, schedule: PrefixAttentionSchedule) -> jax.Array:
    """With start=2, end=6, total=10, the output will be:
    1  1  4/5 3/5 2/5 1/5 0  0  0  0
           ^              ^
         start           end
    `start` (inclusive) is where the chunk starts being allowed to change. `end` (exclusive) is where the chunk stops
    paying attention to the prefix. if start == 0, then the entire chunk is allowed to change. if end == total, then the
    entire prefix is attended to.

    `end` takes precedence over `start` in the sense that, if `end < start`, then `start` is pushed down to `end`. Thus,
    if `end` is 0, then the entire prefix will always be ignored.
    """
    start = jnp.minimum(start, end)
    if schedule == "ones":
        w = jnp.ones(total)
    elif schedule == "zeros":
        w = (jnp.arange(total) < start).astype(jnp.float32)
    elif schedule == "linear" or schedule == "exp":
        w = jnp.clip((start - 1 - jnp.arange(total)) / (end - start + 1) + 1, 0, 1)
        if schedule == "exp":
            w = w * jnp.expm1(w) / (jnp.e - 1)
    else:
        raise ValueError(f"Invalid schedule: {schedule}")
    return jnp.where(jnp.arange(total) >= end, 0, w)


def pinv_corrected_velocity(
    velocity_fn: Callable[[jax.Array, jax.Array], jax.Array],
    x_t: jax.Array,  # [batch, horizon, action_dim]
    y: jax.Array,  # previous action chunk, [batch, horizon, action_dim]
    t: jax.Array,
    inference_delay: int | jax.Array,
    prefix_attention_horizon: int | jax.Array,
    prefix_attention_schedule: PrefixAttentionSchedule,
    max_guidance_weight: float | jax.Array,
) -> jax.Array:
    def denoiser(x_t):
        v_t = velocity_fn(x_t, t)
        return x_t + v_t * (1 - t), v_t

    x_1, vjp_fun, v_t = jax.vjp(denoiser, x_t, has_aux=True)
    weights = get_prefix_weights(inference_delay, prefix_attention_horizon, x_t.shape[-2], prefix_attention_schedule)
    error = (y - x_1) * weights[:, None]
    pinv_correction = vjp_fun(error)[0]
    # constants from paper
    inv_r2 = (t**2 + (1 - t) ** 2) / ((1 - t) ** 2)
    c = jnp.nan_to_num((1 - t) / t, posinf=max_guidance_weight)
    guidance_weight = jnp.minimum(c * inv_r2, max_guidance_weight)
    return v_t + guidance_weight * pinv_correction
