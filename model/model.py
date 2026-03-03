# ruff: noqa: F401, E402
from transformers import PretrainedConfig


class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

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
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union
from transformers import PreTrainedModel, GenerationMixin, PreTrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

class RMSNorm(nn.Module):
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
    

# RoPE旋转位置编码 && YaRN扩展
def precompute_rope_cis(
        dim:int, 
        end:int=int(32*1024), 
        rope_base:float=1e6,
        rope_scaling:Optional[dict]=None
):
    
    # ===================== 步骤1：计算原始RoPE频率（基础公式） =====================
    # 1. torch.arange(0, dim, 2)[: (dim // 2)]：生成0,2,4...dim-2（两两分组的索引）
    # 2. rope_base ** (索引/float(dim))：计算10000^(2i/dim)（RoPE核心频率公式）
    # 3. 1.0 / 结果：得到每个二维组的原始频率θ_i
    freqs = 1.0 / (rope_base ** (torch.arange(0,dim,2)[:dim//2].float()/dim))

    # ===================== 步骤2：YaRN频率缩放（核心扩展） =====================
    # （分段 + 幂次）缩放，低频率组（影响短文本）缩放更保守，
    # 高频率组（影响长文本）大幅缩放，适合中等幅度扩展（比如 2048→8192）
    if rope_scaling is not None:
        # 读取YaRN配置参数
        orig_max, factor, beta_fast, beta_slow = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 4),
            rope_scaling.get("beta_fast", 4),
            rope_scaling.get("beta_slow", 1),
        )
        # 只有当目标长度>原始长度时，才触发YaRN缩放
        if end / orig_max > 1.0:
            # 2.1 计算临界索引
            # 逻辑：找到第一个“频率周期>original_max”的分组索引
            # 频率周期 = 2π/θ_i → 周期越大，频率越低（转得越慢）
            # 作用：区分“低频率组（周期>原始长度）”和“高频率组（周期<原始长度）”
            corr_dim = next((i for i in range(0,dim//2) if 2*math.pi/freqs[i] > orig_max), dim//2)
            # 没找到就设为dim//2（所有组都缩放）

            # 2.2 生成幂次渐变的beta值（从beta_slow到beta_fast）
            # power：0→1的线性序列（0, 1/(dim//2-1), 2/(dim//2-1)...1）
            power = torch.arange(0, dim // 2, device=freqs.device) / max(1, (dim // 2 - 1)) # 避免除0
            # beta：随power从beta_slow（1.0）渐变到beta_fast（4.0）
            beta = beta_slow + (beta_fast - beta_slow) * power
            
            # 2.3 应用分段幂次缩放
            scale = torch.where(
                # 条件：当前分组索引 < corr_dim（低频率组）
                torch.arange(0, dim // 2, device = freqs.device) < corr_dim,
                # 低频率组缩放公式（保守，保短文本精度）：(β*f -β +1)/(β*f)
                (beta * factor - beta + 1) / (beta * factor),
                # 高频率组大幅缩放
                1.0 / factor 
            )

            freqs = freqs * scale
    
    # ===================== 步骤3：计算所有位置的旋转角度 =====================
    t = torch.arange(end, device=freqs.device)
    # torch.outer(t, freqs)：外积→shape=[end, dim//2]，每个位置×每个频率=旋转角度
    freqs = torch.outer(t, freqs).float()

    # ===================== 步骤4：生成cos/sin并补全维度 =====================
    # 因为freqs是[end, dim//2]，需要补全到dim维（两两分组还原）
    # torch.cat([cos,fcos])：把dim//2扩展到dim（每个频率的cos值复制一次）
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim = -1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim = -1)

    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(
    q,  # Q张量，shape=[batch_size, seq_len, num_heads, head_dim]
    k,  # K张量，shape=[batch_size, seq_len, num_kv_heads, head_dim]
    cos,  # precompute_freqs输出的cos，shape=[end, head_dim]
    sin,  # precompute_freqs输出的sin，shape=[end, head_dim]
    unsqueeze_dim=1  # 扩展维度的位置（匹配Q/K的head维度）
):
    
    # 辅助函数：旋转半维（数学技巧，替代手动拆分分组）
    def rotate_half(x):
        # x.shape[-1]//2：取每个头维度的一半（比如128→64）
        # -x[..., 64:]：后64维取负；x[..., :64]：前64维保留
        # 拼接后等价于二维旋转公式，代码更高效
        return torch.cat([-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]], dim=-1)
    #头尾两两配对进行旋转位置编码 （核心公式：x' = x*cos + rotate_half(x)*sin）
    #x1′=x1⋅cosθ−x2⋅sinθ  
    #x2′=x1⋅sinθ+x2⋅cosθ
    #x' = (x1+x2)*cos + (-x2+x1)*sin
    # cos.unsqueeze(unsqueeze_dim)：扩展维度→[8192, 1, 128]（匹配QV的shape）
    q_embed = q * cos.unsqueeze(unsqueeze_dim) + rotate_half(q) * sin.unsqueeze(unsqueeze_dim)
    k_embed = k * cos.unsqueeze(unsqueeze_dim) + rotate_half(k) * sin.unsqueeze(unsqueeze_dim) 

    return q_embed, k_embed

def repeat_kv(x: torch.tensor, n_rep: int) -> torch.tensor:
    bs, seq_len, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
    # 步骤3.1：插入新维度 → 形状变化：[2,1024,8,128] → [2,1024,8,1,128]
    x[:, :, :, None, :]  
    # 步骤3.2：扩展新维度 → 形状变化：[2,1024,8,1,128] → [2,1024,8,4,128]
    .expand(bs, seq_len, num_key_value_heads, n_rep, head_dim)  
    # 步骤3.3：重塑张量 → 形状变化：[2,1024,8,4,128] → [2,1024,32,128]
    .reshape(bs, seq_len, num_key_value_heads * n_rep, head_dim)  
) 

class Attention(nn.Module):
    def __init__(self, args:MiniMindConfig):
        super().__init__() # 调用父类 nn.Module 的初始化函数

        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is \
        None else args.num_key_value_heads

        assert args.num_attention_heads % self.num_key_value_heads == 0, "num_attention_heads must be devisable by num_key_value_heads"

        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.hidden_size // self.n_local_heads

        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias = False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias = False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias = False)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias = False) #outpot线性层

        self.attn_dropout = nn.Dropout(args.dropout)  # attention的dropout
        self.resid_dropout = nn.Dropout(args.dropout) # residual的dropout
        self.dropout = args.dropout

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attention

    def forward(self,
                x: torch.tensor, # 输入张量：[batch_size, seq_len, hidden_size]
                position_embeddings: Tuple[torch.tensor, torch.tensor], # RoPE的cos和sin：各[seq_len, head_dim]
                past_key_value: Optional[Tuple[torch.tensor, torch.tensor]] = None, # 历史KV缓存：推理时传入
                use_cache=False, # 是否保存当前KV到缓存（推理时设为True）
                attention_mask:Optional[torch.tensor]=None): # 注意力掩码：[batch_size, seq_len]
        bs, seq_len, _ = x.shape

        # Q/K/V 投影：将输入从hidden_size映射到 头数×单头维度
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # 拆分多头注意力 重塑维度：[batch_size, seq_len, 头数, 单头维度]（方便后续按头计算）
        xq = xq.view(bs, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bs, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bs, seq_len, self.n_local_kv_heads, self.head_dim)
        # q/k 应用旋转位置编码RoPE
        cos,sin = position_embeddings
        xq,xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])
        
        # 如果传入了历史KV缓存（推理时的前一轮KV），则拼接当前KV
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0],xk],dim=1)
            xv = torch.cat([past_key_value[1],xv],dim=1)
        past_kv = (xk, xv) if use_cache else None

        # 维度转置 + KV头复制（让KV头数匹配Q头数）
        xq, xk, xv = (
            xq.transpose(1,2),# Q：[bsz, n_local_heads, seq_len, head_dim]（头数维度提前，方便按头计算）
            repeat_kv(xk,self.n_rep).transpose(1,2),
            repeat_kv(xv,self.n_rep).transpose(1,2)
        )
        # 进行attention计算
        if self.flash and (seq_len > 1) and (past_key_value is None) and (attention_mask is None or torch.all(attention_mask==1)):
            # FlashAttention：PyTorch内置的高效注意力实现，自动处理缩放、softmax、dropout、因果掩码
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                dropout_p=self.dropout if self.training else 0.0,# 训练时dropout，推理时关闭
                is_causal=True)# 启用因果掩码（防止看到未来token）
        else:
            # 1. 计算注意力分数：Q·K^T / √(head_dim)（缩放防止分数过大）
            scores = (xq@xk.transpose(-2,-1)) / math.sqrt(self.head_dim) # [bsz, heads, seq_len, seq_len]
            # 2. 应用因果掩码（上三角置为-∞，softmax后为0，防止关注未来token）
            scores[:,:,:,-seq_len:] += torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device = scores.device),
                diagonal=1
            )
            # 3. 应用注意力掩码（比如padding部分置为-∞）
            if attention_mask is not None:
                # 扩展掩码维度：[bsz, 1, 1, seq_len]（匹配scores的维度）
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9 # 掩码为0的位置置为-∞
                scores += extended_attention_mask
            # 4. softmax归一化（转float防止精度问题，再转回原类型）
            scores = F.softmax(scores.float(),dim=-1).typed_as(xq)
            # 5. 注意力分数dropout
            scores = self.attn_dropout(scores)
            # 6. 注意力加权求和：分数 × V
            output = scores @ xv  # [bsz, heads, seq_len, head_dim]
        # 转置+重塑维度：[bsz, heads, seq_len, head_dim] → [bsz, seq_len, heads×head_dim]
        output = output.transpose(1,2).reshape(bs,seq_len,-1)
        # 输出投影（还原到hidden_size） + residual dropout
        output = self.resid_dropout(self.o_proj(output))
        
        return output, past_kv



        








            



    
    

    



