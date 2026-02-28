from transformers import Optional, PretrainedConfig
import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Union
from torch import functional as F
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast


class MokioMindConfig(PretrainedConfig):
    model_type = "mokiomind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.1,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 4,
                "beta_slow": 1,
                "factor": 4,
                "original_max_position_embeddings": 2048,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )


# inherent from nn.Module class
class RMSNorm(nn.Module):
    # use __init__ to initialize the model parameters and configurations
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    # _norm
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    # forward function to define the forward pass of the model
    def forward(self, x):
        return self.weight * self.norm(x.float()).type_as(x) * x


def precompute_rope_cis(
    dim: int,
    end: int = int(32 * 1024),
    rope_base: int = 1e6,
    rope_scaling: Optional[dict] = None,
):
    # RoPE function
    freqs = 1.0 / (
        rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
    )

    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 4),
            rope_scaling.get("beta_fast", 4),
            rope_scaling.get("beta_slow", 1),
        )

        if end / orig_max > 1.0:
            # calculate corr_dim
            corr_dim = next(
                (
                    i
                    for i in range(dim // 2)
                    if 2 * math.pi / freqs[i] > orig_max
                ),
                dim // 2,
            )  # find the smallest i such that 2 * pi / freqs[i] > orig_max, if not found return dim // 2
            # **why use next()?** because we only need the first i that satisfies the condition

            # calculate power
            power = torch.arrange(
                0, dim // 2, device=freqs.device
            ).float() / max(dim // 2 - 1, 1)

            # calculate beta
            beta = beta_slow + (beta_fast - beta_slow) * power

            # calculate scale
            scale = torch.where(
                torch.arange(dim // 2, device=freqs.device) < corr_dim,
                (beta * factor - beta + 1) / (beta * factor),
                1 / factor,
            )

            # apply scale
            freqs = freqs * scale

    # generate position ids, and calculate the outer product of position ids and freqs
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()

    # return the con and sin matraix
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin


# apply YaRN (RoPE) to the input tensor


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    # **why need rotate_half?** because we need to rotate the second half of the input tensor, and keep the first half unchanged
    def rotate_half(x):  # [a,b] -> [-b,a]
        # x.shape[-1]: the last dimension of x, which is the hidden size of the model
        # x[..., x.shape[-1] // 2 :]: the second half of the input tensor
        # x[..., : x.shape[-1] // 2]: the first half of the
        return torch.cat(
            (-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1
        )

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(q) * sin.unsqueeze(unsqueeze_dim)
    )
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(k) * sin.unsqueeze(unsqueeze_dim)
    )
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = (
        x.shape
    )  # [batch_size, seq_len, num_key_value_heads, head_dim]
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: MokioMindConfig):
        super().__init__()
        self.num_key_value_heads = (
            args.num_key_value_heads
            if args.num_key_value_heads is not None
            else args.num_attention_heads
        )

        assert args.num_attention_heads % self.num_key_value_heads == 0, (
            "num_attention_heads must be divisible by num_key_value_heads"
        )

        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = (
            self.n_local_heads // self.num_key_value_heads
        )  # floor division: divides two numbers and rounds the result down to the nearest integer
        self.head_dim = args.hidden_size // args.num_attention_heads

        self.q_proj = nn.Linear(
            args.hidden_size, self.n_local_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            args.hidden_size,
            self.n_local_kv_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            args.hidden_size,
            self.n_local_kv_heads * self.head_dim,
            bias=False,
        )
        self.out_proj = nn.Linear(
            self.n_local_heads * self.head_dim, args.hidden_size, bias=False
        )

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        self.flash = (
            hasattr(torch.nn.functional, "scaled_dot_product_attention")
            and args.flash_attention
        )

    def forward(
        self,
        x: torch.Tensor,
        position_embedding: tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        # projection: calculate the query, key and value matrices by applying linear transformations to the input tensor
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # divide the input tensor into multiple heads using .view()
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # apply RoPE to the query
        cos, sin = position_embedding
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])

        # for key and value, use .repeat(), meanwhile d ote kv cache for inference
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        pass_kv = (xk, xv) if use_cache else None

        xq, xk, xv = (
            xq.transpose(
                1, 2
            ),  # [bsz, n_local_heads, seq_len, head_dim] before transpose, [bsz, seq_len, n_local_heads, head_dim] after transpose
            # why need transpose? because we need to calculate the attention scores by multiplying the query and key matrices, and the query and key matrices need to be in the shape of [batch_size, num_heads, seq_len, head_dim] to perform the pytroch's matrix multiplication
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2),
        )
        # calculate the attention scores : q@k^T/sqrt(head_dim)
        if (
            self.flash
            and seq_len > 1
            and attention_mask is None
            or torch.all(attention_mask == 1)
        ):
            # torch.all(attention_mask==1) means no attention mask, because the attention mask is usually a binary matrix where 1 indicates that the token can attend to the other token, and 0 indicates that the token cannot attend to the other token. if all elements in the attention mask are 1, it means that all tokens can attend to each other, which is equivalent to no attention mask.
            attn_mask = (
                None
                if attention_mask is None
                else attention_mask.view(
                    bsz, 1, 1, -1
                )  # reshape to [bsz, n_heads, seq_len]
                .expand(
                    bsz, self.n_local_heads, seq_len, -1
                )  # expand to [bsz, n_local_heads, seq_len, seq_len]
                .bool()  # convert to boolean tensor
            )
            output = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )  # pytorch's build-in attention function, faster
        else:  # however, let's also try to implement the attention calculation by ourselves
            scores = (xq @ xk.transpose(-1, -2)) / torch.sqrt(
                torch.tensor(self.head_dim, dtype=torch.float32)
            )
            scores = (
                scores
                + torch.triu(
                    torch.full(
                        (seq_len, seq_len), float("-inf"), device=scores.device
                    ),
                    diagonal=1,
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )  # .full((seq_len, seq_len), float("-inf")) creates a matrix of shape [seq_len, seq_len] filled with -inf, and torch.triu(..., diagonal=1) sets the upper triangular part of the matrix (excluding the main diagonal) to -inf, which is used to mask out the future tokens in the attention calculation. unsqueeze(0).unsqueeze(0) adds two dimensions to the matrix to make it compatible with the shape of the attention scores, which has two leading dimensions (batch and head).

            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(
                    2
                )  # reshape to [bsz, 1, 1, seq_len]
                extended_attention_mask = (
                    (1.0 - extended_attention_mask) * float("-inf")
                )  # convert the attention mask to a mask that can be added to the attention scores, where 1 in the original attention mask becomes 0 (no masking) and 0 becomes -inf (masking)
                scores = (
                    scores + extended_attention_mask
                )  # add the attention mask to the attention scores

            scores = F.softmax(
                scores.float(), dim=-1
            )  # apply softmax to the attention scores to get the attention weights
            scores = self.attn_dropout(
                scores
            )  # apply dropout to the attention weights
            output = (
                scores @ xv
            )  # calculate the attention output by multiplying the attention weights with the value

        # concatenate the output of all heads and apply a final linear transformation to get the output tensor
        output = output.transpose(1, 2).reshape(
            bsz, seq_len, -1
        )  # reshape the output back to [bsz, seq_len, hidden_size]
        output = self.resid_dropout(self.out_proj(output))  # apply the ResNet
        return output, pass_kv


class FeedForward(nn.Module):
    # initialization
    # elevate the dimension
    # decrease the dimension
    # gate
    # dropout
    # activation function
    def __init__(self, args: MokioMindConfig):
        super().__init__()
        if args.intermediate_size is None:
            intermediate_size = int(
                args.hidden_size * 8 / 3
            )  # 8/3 is a common ratio used in NN with gated activation
            args.intermediate_size = (
                64 * ((intermediate_size + 64 - 1) // 64)
            )  # round up to the nearest multiple of 64 for better performance on GPU

        self.up_proj = nn.Linear(
            args.hidden_size, args.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            args.intermediate_size, args.hidden_size, bias=False
        )
        self.gate_proj = nn.Linear(
            args.hidden_size, args.intermediate_size, bias=False
        )
        self.dropout = nn.Dropout(args.dropout)
        self.act_fn = ACT2FN[args.hidden_act]

    def forward(self, x):
        gated = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(self.down_proj(gated))


class MokioMindBlock(
    nn.Module
):  # this is actually a single transformer block, including a self-attention layer and a feedforward layer, with residual connections and layer normalization
    def __init__(self, Layer_id: int, config: MokioMindConfig):
        super().__init__()
        self.number_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.self_attn = Attention(config)

        self.layer_id = Layer_id
        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = FeedForward(config)

    def forward(
        self,
        hidden_states,
        position_embddings,
        past_key_value=None,
        use_cache=False,
        attention_mask=None,
    ):
        residual = hidden_states
        hidden_states, present_key_values = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embddings,
            past_key_value=past_key_value,
            use_cache=use_cache,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return hidden_states, present_key_values


class MokioMindModel(nn.Module):
    def __init__(self, config: MokioMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size.self.num_hidden_layers = (
            config.vocab_size,
            config.num_hidden_layers,
        )
        self.embeded_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size
        )
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [MokioMindBlock(i, config) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # RoPE precomputation: precompute the RoPE cosine and sine values for the maximum position embeddings, and register them as buffers in the model so that they can be used during the forward pass without being updated during training.
        freq_cos, freq_sin = precompute_rope_cis(
            dim=config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        self.register_buffer(
            "freq_cos", freq_cos, persistent=False
        )  # register_buffer is a method in nn.Module that allows you to register a tensor as a buffer, which means that it will be saved and loaded with the model, but it will not be considered as a model parameter and will not be updated during training. in this case, we register the precomputed RoPE cosine and sine values as buffers, so that they can be used during the forward pass without being updated during training.
        self.register_buffer("freq_sin", freq_sin, persistent=False)

        def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple[torch.Tensor]] = None,
            use_cache: bool = False,
            **kwargs,
        ):
            batch_size, seq_len = input_ids.shape

            if hasattr(past_key_values, "layers"):
                past_key_values = None  # for backward compatibility (hugginface) with old checkpoints

            past_key_values = (
                past_key_values or [None] * len(self.layers)
            )  # if past_key_values is None, create a list of None with the same length as the number of layers, otherwise use the provided past_key_values

            start_pos = (
                past_key_values[0][0].shape[1]
                if past_key_values[0] is not None
                else 0
            )  # calculate the starting position for the current input based on the length of the past key values, if past_key_values is not None, get the shape of the first element (the key) of the first layer's past key values, and use the second dimension (the sequence length) as the starting position, otherwise start from 0 (no key value cache)

            hidden_states = self.dropout(self.embeded_tokens(input_ids))
            position_embddings = (
                self.freq_cos[start_pos : start_pos + seq_len],
                self.freq_sin[start_pos : start_pos + seq_len],
            )
            presents = []

            for layer_idx, (layer, past_key_value) in enumerate(
                zip(self.layers, past_key_values)
            ):  # iterate through each transformer block (named as MokioMindBlock) and its corresponding past key values, and apply the forward pass of each block to the hidden states, while also passing the position embeddings and attention mask. collect the present key values for each layer in a list called presents, which will be used for caching during inference.
                hidden_states, present = layer(
                    hidden_states,
                    position_embddings,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    attention_mask=attention_mask,
                )
                presents.append(
                    present
                )  # collect the present key values for each layer, which will be used for caching during inference

            hidden_states = self.norm(
                hidden_states
            )  # apply layer normalization to the final hidden states after all transformer blocks
            return hidden_states, presents


class MokioMindForCausalLM(PreTrainedModel, GenerationMixin):
    # PreTrainedModel: a HugginFace's base class for all models in the transformers library, which provides common methods for loading and saving models, and handling model configurations. GenerationMixin: a mixin class that provides methods for generating text using the model, such as greedy decoding, beam search, etc.
    # GenerationMixin is a HugginFace's mixin class that provides methods for generating text using the model, such as greedy decoding, beam search, etc. By inheriting from both PreTrainedModel and GenerationMixin, MokioMindForCausalLM can leverage the functionalities provided by both classes, allowing it to be easily loaded and saved using the PreTrainedModel's methods, and to generate text using the GenerationMixin's methods.
    config_class = MokioMindConfig

    def __init__(self, config: MokioMindConfig):
        self.config = config
        super().__init__(config)
        self.model = MokioMindModel(config)
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )  # a linear layer that maps the hidden states to the vocabulary size, which will be used to generate the logits for each token in the vocabulary during the forward pass. The bias is set to False because we want to tie the weights of the lm_head with the input embedding layer, which is a common technique in language modeling to reduce the number of parameters and improve performance. By tying the weights, we ensure that the same weights are used for both the input embeddings and the output logits, which can help the model learn better representations and generate more coherent text.

        # tie weights: tie (share) the weights of the lm_head with the input embedding layer, which is a common technique in language modeling to reduce the number of parameters and improve performance. By tying the weights, we ensure that the same weights are used for both the input embeddings and the output logits, which can help the model learn better representations and generate more coherent text.
        self.model.embeded_tokens.weight

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: bool = False,
        logits_to_keep: Union[
            int, torch.Tensor
        ] = 0,  # for efficient inference with large vocabularies, we can keep only the top-k logits for each token, where k is specified by logits_to_keep. This can help reduce the computational cost and memory usage during inference, especially when the vocabulary size is large. If logits_to_keep is set to 0, it means that we keep all logits without any filtering.
        **args,
    ):
        hidden_states, past_key_values = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args,
        )
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )  # if logits_to_keep is an integer, we create a slice object that keeps the last #N=logits_to_keep elements of the logits tensor, which corresponds to the top-k logits. If logits_to_keep is a tensor, we use it directly as the indices to keep in the logits tensor.
        logits = self.lm_head(
            hidden_states
        )[
            :, slice_indices, :
        ]  # apply the lm_head linear layer to the hidden states to get the logits for each token in the vocabulary, and then slice the logits tensor to keep only the specified indices (top-k logits) for efficient inference with large vocabularies.

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
        )
