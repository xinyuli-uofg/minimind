from transformers import Optional, PretrainedConfig


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

import torch
import torch.nn as nn
import math

# inherent from nn.Module class
class RMSNorm(nn.Module):
# use __init__ to initialize the model parameters and configurations
    def __init__(self, dim:int, eps:float=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

# _norm
    def _norm(self, x):
        return torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)

# forward function to define the forward pass of the model
    def forward(self,x):
        return self.weight*self.norm(x.float()).type_as(x)*x
    


def precompute_rope_cis(dim:int, end:int=int(32*1024), rope_base:int=1e6, rope_scaling:Optional[dict]=None):
    # RoPE function
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2).float() / dim))

    if rope_scaling is not None: 
        orig_max, factor, beta_fast, beta_slow = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 4),
            rope_scaling.get("beta_fast", 4),
            rope_scaling.get("beta_slow", 1),
        )

        # calculate corr_dim
        corr_dim = next(
                    (i for i in range(dim // 2) if 2 * math.pi / freqs[i] > orig_max),
                    dim // 2,
                ) # find the smallest i such that 2 * pi / freqs[i] > orig_max, if not found return dim // 2
                # **why use next()?** because we only need the first i that satisfies the condition

        # calculate power
        power = torch.arrange(0,dim//2,device=freqs.device).float()/max(dim//2-1,1)

        # calculate beta
        beta = beta_slow+(beta_fast-beta_slow)*power

        # calculate scale 
        scale = torch.where(
            torch.arange(dim//2,device=freqs.device)<corr_dim,
            (beta*factor-beta+1)/(beta*factor),
            1/factor
        )

        # apply scale
        freqs = freqs*scale
    
    # generate position ids, and calculate the outer product of position ids and freqs
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()

    # return the con and sin matraix
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin

# apply YaRN (RoPE) to the input tensor

def apply_rotary_pos_emb(q,k,cos,sin,unsqueeze_dim=1):
    # **why need rotate_half?** because we need to rotate the second half of the input tensor, and keep the first half unchanged
    def rotate_half(x): # [a,b] -> [-b,a]
        # x.shape[-1]: the last dimension of x, which is the hidden size of the model
        # x[..., x.shape[-1] // 2 :]: the second half of the input tensor
        # x[..., : x.shape[-1] // 2]: the first half of the
        return torch.cat(
            (-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1
        )
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed
    
