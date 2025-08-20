import dataclasses
import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

from functools import partial
logger = logging.getLogger("openpi")

PALIGEMMA_EOS_TOKEN = 1


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


def put_along_last_axis(arr, indices, values):
    """Like np.put_along_axis(..., axis=-1), since jax is missing it."""
    assert arr.ndim == indices.ndim == values.ndim, (arr.ndim, indices.ndim, values.ndim)
    onehot = jax.nn.one_hot(indices, arr.shape[-1], dtype=values.dtype)
    put_mask = jnp.einsum("...i,...in->...n", jnp.ones(values.shape, jnp.int32), onehot)
    put_values = jnp.einsum("...i,...in->...n", values, onehot)
    return jnp.where(put_mask, put_values, arr)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


@dataclasses.dataclass(frozen=True)
class Pi0CTPConfig(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 48

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0_CTP

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0CTP":
        return Pi0CTP(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                token_ar_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.bool_),
                token_loss_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.bool_),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)


class Pi0CTP(_model.BaseModel):
    def __init__(self, config: Pi0CTPConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
            )
        )
        llm.lazy_init(rngs=rngs, method="init")
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, "b s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask.append(0 * input_mask[-1])
        
        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask.append(obs.token_ar_mask)
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.concatenate(ar_mask, axis=1)
        ar_mask = jnp.array(ar_mask, dtype=jnp.bool)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, "b s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # add a single state token
        state_token = self.state_proj(obs.state)[:, None, :]
        tokens.append(state_token)
        input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
        # image/language inputs do not attend to state or actions
        ar_mask += [True]

        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        # mix timestep + action information using an MLP
        action_tokens = self.action_in_proj(noisy_actions)
        time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
        action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
        action_time_tokens = self.action_time_mlp_in(action_time_tokens)
        action_time_tokens = nnx.swish(action_time_tokens)
        action_time_tokens = self.action_time_mlp_out(action_time_tokens)
        tokens.append(action_time_tokens)
        input_mask.append(jnp.ones(action_time_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        ar_mask = jnp.broadcast_to(ar_mask, (tokens.shape[0], tokens.shape[1]))
        return tokens, input_mask, ar_mask


    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> tuple[at.Float[at.Array, "*b ah"], dict[str, at.Array]]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001 # 1에 가까운 time이 더 많이 샘플링 되는 분포
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions # time이 1에 가깝게 더 많이 샘플링 되기 때문에, noisy action을 더 많이 사용하게 됨.
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=1)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions
        )

        txt_targets = jax.nn.one_hot(
            observation.tokenized_prompt[:, 1:],
            _gemma.PALIGEMMA_VOCAB_SIZE,
        )

        txt_logits = self.PaliGemma.llm(
            prefix_out[:, 768:-1], 
            method="embedder_decode"
        )
        txt_logp = jax.nn.log_softmax(txt_logits, axis=-1)

        txt_token_loss = jnp.sum(txt_targets * txt_logp, axis=-1)
        txt_loss_mask = observation.token_loss_mask[:, 1:]
        txt_loss = (
            -jnp.sum(txt_token_loss * txt_loss_mask, axis=-1) /
            jnp.clip(jnp.sum(txt_loss_mask, axis=-1), 1)
            )

        #* test for text tokenizer and debugging
        # top_txt_logp = jnp.argmax(txt_logp, axis=-1)
    
        # import sentencepiece
        # import openpi.shared.download as download

        # path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        # with path.open("rb") as f:
        #     self.tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        # jax.debug.breakpoint()
        #* test code end
        
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        action_loss = jnp.mean(jnp.square(v_t - u_t), axis=(-2, -1))

        loss = action_loss + txt_loss

        info = {
            "loss": loss,
            "action_loss": action_loss,
            "txt_loss": txt_loss,
        }

        return loss, info

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
        
        # import sentencepiece
        # import openpi.shared.download as download
        # prefix_out, suffix_out = _

        # txt_logits = self.PaliGemma.llm(
        #     prefix_out[:, 768:, :], 
        #     method="embedder_decode"
        # )
        # txt_logp = jax.nn.log_softmax(txt_logits, axis=-1)
        # top_txt_logp = jnp.argmax(txt_logp, axis=-1)

        # path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        # with path.open("rb") as f:
        #     self.tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        # jax.debug.breakpoint()

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens], mask=full_attn_mask, positions=positions, kv_cache=kv_cache
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    
    def predict_bbox(self, 
                     rng: at.KeyArrayLike, 
                     observation: _model.Observation,
                     max_decoding_steps: int = 48,
                     ):
        observation = _model.preprocess_observation(None, observation, train=False, image_keys=list(observation.images.keys()))
        before_padding_prompt = observation.tokenized_prompt

        first_one_indices = jnp.argmax(observation.token_ar_mask, axis=-1)
        padding_mask = jnp.arange(observation.token_ar_mask.shape[-1]) >= first_one_indices[..., jnp.newaxis]
        # padding suffix to 0
        observation = dataclasses.replace(
            observation, 
            tokenized_prompt=jnp.where(padding_mask, 0, observation.tokenized_prompt),
            tokenized_prompt_mask=jnp.logical_not(padding_mask),
        )
        # embed inputs
        prefix_token_embeddings, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        # first fill KV cache with a forward pass of the prefix (img + txt prefix)
        (pre_logit, _), pre_kv_cache = self.PaliGemma.llm(
            [prefix_token_embeddings, None], mask=prefix_attn_mask, positions=prefix_positions
        )
        
        # import sentencepiece
        # import openpi.shared.download as download
        # prefix_out = pre_logit

        # txt_logits = self.PaliGemma.llm(
        #     prefix_out[:, 768:, :], 
        #     method="embedder_decode"
        # )
        # txt_logp = jax.nn.log_softmax(txt_logits, axis=-1)
        # top_txt_logp = jnp.argmax(txt_logp, axis=-1)

        # path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        # with path.open("rb") as f:
        #     self.tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        # jax.debug.breakpoint()

        eop_indices = prefix_positions[:, -1]
        eop_pre_logit = jnp.take_along_axis(pre_logit, eop_indices[:, None, None], axis=1)
        eop_logit = self.PaliGemma.llm(eop_pre_logit, method="embedder_decode")

        valid_tokens = jnp.array([108])
        valid_mask = jnp.full((1, 1, eop_logit.shape[-1]), -jnp.inf)
        valid_mask = valid_mask.at[:, :, valid_tokens].set(0)
        eop_logit = eop_logit + valid_mask

        token = jnp.argmax(eop_logit, axis=-1)
        has_eos = jnp.any(token==PALIGEMMA_EOS_TOKEN, axis=1)
        all_eos = jnp.all(has_eos)
        output_tokens = jnp.zeros((eop_logit.shape[0], max_decoding_steps), dtype=token.dtype)

        # prefix_mask가 False인 부분에 대해 pre_kv_cache 값을 0으로 만들기
        # prefix_mask: (1, 816) -> (1, 1, 816, 1, 1)로 브로드캐스팅
        prefix_mask_broadcasted = prefix_mask[:, None, :, None, None]
        
        
        
        

        #! Test for attention mask
        # kv_cache = jax.tree.map(
        #     lambda x: jnp.pad(x, ((0, 0), (0, 0), (0, max_decoding_steps), (0, 0), (0, 0))), pre_kv_cache) # (18, 1, 864, 1, 256)
        
        # prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=1)
        # attn_mask = jnp.pad(prefix_attn_mask,
        #                     ((0, 0), (0, 0), (0, max_decoding_steps + 1)))
        # attn_mask = attn_mask.at[:, :, -1].set(True) # (1, 1, 865)

        

        

        #* test code
        # padding kv_cache and attn_mask
        kv_cache = jax.tree.map(
            lambda x: jnp.where(prefix_mask_broadcasted, x, 0), pre_kv_cache)
        attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=1)
        attn_mask = jnp.pad(attn_mask, ((0, 0), (0, 0), (0, 1)))
        attn_mask = attn_mask.at[:, :, -1].set(True)
        last_true_position = jnp.argmax(prefix_positions, axis=-1)[0]


        # prefix_mask가 True인 값만 추출하여 새로운 kv_cache 생성
        # prefix_mask: (1, 816) -> (816,)로 squeeze
        # prefix_mask_flat = prefix_mask.squeeze()  # (816,)
        # true_count = jnp.sum(prefix_mask_flat)

        # kv_cache = jax.tree.map(
        #     lambda x: x[:, :, :true_count, :, :], pre_kv_cache)
        # attn_mask = prefix_mask[:, :true_count]
        # attn_mask = attn_mask[:, None, :]
        
        # kv_cache = jax.tree.map(
        #     lambda x: jnp.compress(prefix_mask_flat, x, axis=2), pre_kv_cache)  # (18, 1, N, 1, 256)
        
        # attn_mask도 동일하게 처리
        # attn_mask = jnp.compress(prefix_mask_flat, prefix_mask.squeeze(), axis=1)  # (1, N)
        # attn_mask = attn_mask[:, None, :]  # (1, 1, N)
        
        # print(f"Original shape: {pre_kv_cache[0].shape}")
        # print(f"Compressed shape: {kv_cache[0].shape}")
        # print(f"True count: {jnp.sum(prefix_mask_flat)}")

    


        # import sentencepiece
        # import openpi.shared.download as download

        # path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        # with path.open("rb") as f:
        #     self.tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        # jax.debug.breakpoint()

        # batch_size = prefix_mask.shape[0]
        # suffix_ar_mask = jnp.ones((batch_size, max_decoding_steps + 1), dtype=jnp.bool_)  # causal attention for suffix
        
        # # 전체 attention mask 구성: prefix (bidirectional) + suffix (causal)
        # full_ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=1)
        # full_input_mask = jnp.concatenate([prefix_mask, jnp.ones((batch_size, max_decoding_steps + 1), dtype=jnp.bool_)], axis=1)
        # full_attn_mask = make_attn_mask(full_input_mask, full_ar_mask)

        #! Test finished



        def _wrap_cache(cache_appended, step):
            # KV cache는 이미 패딩되어 있으므로, 새로운 값을 올바른 위치에 업데이트
            # cache_appended는 (l, b, t+1, k, h)이고, kv_cache는 (l, b, t+max_decoding_steps, k, h)
            new_value = cache_appended[:, :, -1]  # 마지막 토큰의 KV cache
            # prefix_mask.shape[1] + 1 + step 위치에 새로운 값을 삽입
            # insert_pos = prefix_mask.shape[1] + 1 + step
            insert_pos = last_true_position + 1 + step
            return jax.lax.dynamic_update_index_in_dim(
                cache_appended[:, :, :-1],  # 마지막 차원 제거 (t+1 -> t)
                new_value,
                insert_pos,
                axis=2
            )
        # import sentencepiece
        # import openpi.shared.download as download

        # path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        # with path.open("rb") as f:
        #     self.tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        # jax.debug.breakpoint()
        def decode_step(carry):
            last_logit, output_tokens, kv_cache, attn_mask, _, step = carry

            token = jnp.argmax(last_logit, axis=-1)
            token = jnp.where(
                step==0,
                jnp.full_like(token, 108),
                token
            )
            output_tokens = put_along_last_axis(
                output_tokens, jnp.broadcast_to(step, (token.shape[0], 1)), token
            )
            has_eos = jnp.any(token == PALIGEMMA_EOS_TOKEN, axis=1)
            all_eos = jnp.all(has_eos)

            token_embedding = self.PaliGemma.llm(token, method="embed")
            positions = prefix_positions[:, [-1]] + step + 1

            # import sentencepiece
            # import openpi.shared.download as download

            # path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
            # with path.open("rb") as f:
            #     self.tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

            # jax.debug.breakpoint()
            
            (last_pre_logit, _), kv_cache_appended = self.PaliGemma.llm(
                [token_embedding, None], mask=attn_mask, positions=positions, kv_cache=kv_cache
            )
            last_logit = self.PaliGemma.llm(last_pre_logit, method="embedder_decode")
            kv_cache = jax.tree.map(
                lambda x: _wrap_cache(x, step),
                kv_cache_appended,
            )
            attn_mask = attn_mask.at[:, :, last_true_position + 1 + step].set(True)

            return last_logit, output_tokens, kv_cache, attn_mask, all_eos, step + 1

        def decode_cond(carry):
            _, _, _, _, all_eos, step = carry
            return (~all_eos) & (step < max_decoding_steps)
        
        _, suffix_txt_tokens, kv_cache, prefix_attn_mask, _, _ = \
            jax.lax.while_loop(
                decode_cond, decode_step,
                (eop_logit, output_tokens, kv_cache, attn_mask, all_eos, 0),
                )
            #* test code
            # jax.lax.while_loop(
            #     decode_cond, decode_step,
            #     (eop_logit, output_tokens, kv_cache, full_attn_mask, all_eos, 0),
            #     )
        

        return suffix_txt_tokens, kv_cache, prefix_attn_mask
        
            
        


    def sample_actions_with_bbox(self, 
                                  rng: at.KeyArrayLike,
                                  observation: _model.Observation,
                                  *,
                                  _kv_cache,
                                  _prefix_attn_mask,
                                  num_steps: int | at.Int[at.Array, ""] = 10,
                                  ):
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
        attn_mask = _prefix_attn_mask[:, :, :-1]
        # change last true token to false
        # find last true token
        last_true_position = jnp.argmax(jnp.cumsum(attn_mask, axis=-1)-1, axis=-1)
        attn_mask = attn_mask.at[:, :, last_true_position].set(False)
        kv_cache = _kv_cache

        # kv_cache = jax.tree.map(
            # lambda x: x[:, :, :-1, :, :], _kv_cache)



        # import sentencepiece
        # import openpi.shared.download as download

        # path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        # with path.open("rb") as f:
        #     self.tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())
        # jax.debug.breakpoint()

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_mask = einops.repeat(attn_mask, "b s p -> (b s) p") # (1, 816)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                attn_mask.shape[2] + suffix_tokens.shape[1],
            )
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens], mask=full_attn_mask, positions=positions, kv_cache=kv_cache
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            # v_t = 1.0
            # jax.debug.breakpoint()

            return x_t + dt * v_t, time + dt

            
        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        

        return x_0

    