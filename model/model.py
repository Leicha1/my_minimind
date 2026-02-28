from transformers import PretrainedConfig


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

import math
import torch
import torch.nn.init as init
import totch.nn.function as F
from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union
from transformers import PreTrainedModel, GenerationMixin, PreTrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super.__init__()
        self.eps = eps
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(dim))
        
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim = True) + self.eps)
    """
    x.pow(2)：对输入张量 x 的每个元素做平方运算（计算元素的平方值）;
    .mean(-1, keepdim=True);
    -1:按最后一维计算均值（比如 x 形状是 [batch, seq_len, dim]，则对每个 dim 维度求均值，得到 [batch, seq_len, 1]);
    keepdim=True: 保持维度数不变，避免广播维度不匹配;
    + self.eps:加上极小值 eps,防止后续开方时分母为 0;
    torch.rsqrt(...)：计算 “倒数平方根” 1 / sqrt(x)，等价于 1 / torch.sqrt(...)，是更高效的实现；
    x * ...：将原输入 x 乘以上述倒数平方根，完成 “均方根归一化”。
    """

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)
    
    

    



