# ruff: noqa: E402
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
        flash_attn: bool = True,
        ####################################################
        # Here are the specific configurations of MOE
        # When use_moe is false, the following is invalid
        ####################################################
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = 'softmax',
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs
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
        # 外推长度 = factor * original_max_position_embeddings = 32768
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        self.flash_attn = flash_attn
        ####################################################
        # Here are the specific configurations of MOE
        # When use_moe is false, the following is invalid
        ####################################################
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家
        self.scoring_func = scoring_func  # 评分函数，默认为'softmax'
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率

import math
import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
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
def precompute_freqs_cis(
    dim: int,  # RoPE位置编码的维度（需为偶数，通常等于模型hidden_size）
    end: int = int(32 * 1024),  # 预计算的最大序列长度（覆盖训练/推理的最长文本）
    rope_base: float = 1e6,  # RoPE的基础常数（原始RoPE论文默认1e6）
    rope_scaling: Optional[dict] = None  # YaRN缩放配置（None则使用标准RoPE）
):
    # ========== 核心行1：计算标准RoPE的基础频率 + 初始化注意力因子 ==========
    # 1. torch.arange(0, dim, 2)[: (dim // 2)]：取0到dim的偶数索引（步长2），截断到dim//2个（确保成对）
    # 2. / dim：将索引归一化到[0,1)区间
    # 3. rope_base ** (...)：计算RoPE的频率分母项 (base^(2i/dim))
    # 4. 1.0 / (...)：得到基础频率 freqs = 1 / (base^(2i/dim))，形状为 [dim//2]
    # 5. attn_factor：注意力缩放因子，初始为1.0（YaRN中可调整）
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0

    # ========== YaRN缩放逻辑（解决长文本RoPE退化问题） ==========
    if rope_scaling is not None:  # 若配置了YaRN缩放
        # 解析YaRN配置参数（默认值适配主流场景）
        # orig_max：原始模型训练的最大序列长度（默认2048）
        # factor：RoPE频率缩放因子（默认16）
        # beta_fast/beta_slow：YaRN的快慢衰减系数（控制不同维度的缩放程度）
        # attn_factor：注意力分数的缩放因子（覆盖初始的1.0）
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), 
            rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), 
            rope_scaling.get("beta_slow", 1.0), 
            rope_scaling.get("attention_factor", 1.0)
        )

        # 仅当目标序列长度 > 原始最大长度时，才启用YaRN缩放（否则用标准RoPE）
        if end / orig_max > 1.0:
            # YaRN核心公式：f'(i) = f(i) * [(1-γ) + γ/s]，其中γ∈[0,1]是线性斜坡函数
            # 定义逆维度函数：计算对应beta值的临界维度索引（划分快/慢衰减维度）
            # inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            def inv_dim(b: float):
                numerator = dim * math.log(orig_max / (b * 2 * math.pi))
                denominator = 2 * math.log(rope_base)
                return numerator / denominator
            # 计算快/慢衰减对应的维度边界（low=快衰减起始，high=慢衰减结束）
            # max/min确保边界在合法范围[0, dim//2-1]内
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            
            # 生成线性斜坡ramp：形状[dim//2]，值在0~1之间（控制不同维度的缩放强度）
            # max(high - low, 0.001)避免分母为0
            ramp = torch.clamp(
                (torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001),
                0,  # 下限：0（不缩放）
                1   # 上限：1（最大缩放）
            )
            
            # 应用YaRN频率调整：低频维度（近low）缩放少，高频维度（近high）缩放多
            freqs = freqs * (1 - ramp + ramp / factor)

    # ========== 生成最终的RoPE频率矩阵 ==========
    # 生成时间步（位置索引）：t ∈ [0, end-1]，形状[end]
    t = torch.arange(end, device=freqs.device)
    
    # 计算外积：t（位置） × freqs（维度频率） → 形状[end, dim//2]
    # 含义：每个位置、每个维度对的基础频率值
    freqs = torch.outer(t, freqs).float()
    
    # ========== 生成余弦/正弦矩阵（适配完整dim维度） ==========
    # 1. torch.cos(freqs)：形状[end, dim//2] → 拼接两次 → [end, dim]（成对复制，适配RoPE的偶数维度）
    # 2. * attn_factor：应用注意力缩放因子（调整注意力分数幅度）
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    
    # 正弦矩阵逻辑与余弦完全一致
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    
    # 返回RoPE所需的余弦和正弦频率矩阵（形状均为[end, dim]）
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

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
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

        assert args.num_attention_heads % self.num_key_value_heads == 0, \
            "num_attention_heads must be divisable by num_key_value_heads"

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

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn

    def forward(self,
                x: torch.Tensor, # 输入张量：[batch_size, seq_len, hidden_size]
                position_embeddings: Tuple[torch.Tensor, torch.Tensor], # RoPE的cos和sin：各[seq_len, head_dim]
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, # 历史KV缓存：推理时传入
                use_cache=False, # 是否保存当前KV到缓存（推理时设为True）
                attention_mask:Optional[torch.Tensor]=None): # 注意力掩码：[batch_size, seq_len]
        bs, seq_len, _ = x.shape

        # Q/K/V 投影：将输入从hidden_size映射到 头数×单头维度
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # 拆分多头注意力 重塑维度：[batch_size, seq_len, 头数, 单头维度]（方便后续按头计算）
        xq = xq.view(bs, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bs, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bs, seq_len, self.n_local_kv_heads, self.head_dim)
        # q/k 应用旋转位置编码RoPE
        cos,sin = position_embeddings
        xq,xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        
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
            # scores = F.softmax(scores.float(),dim=-1).typed_as(xq)
            scores = F.softmax(scores.float(), dim=-1).to(dtype=xq.dtype, device=xq.device)
            # 5. 注意力分数dropout
            scores = self.attn_dropout(scores)
            # 6. 注意力加权求和：分数 × V
            output = scores @ xv  # [bsz, heads, seq_len, head_dim]
        # 转置+重塑维度：[bsz, heads, seq_len, head_dim] → [bsz, seq_len, heads×head_dim]
        output = output.transpose(1,2).reshape(bs,seq_len,-1)
        # 输出投影（还原到hidden_size） + residual dropout
        output = self.resid_dropout(self.o_proj(output))
        
        return output, past_kv

class FeedForward(nn.Module):
    def __init__(self, args:MiniMindConfig):
        super().__init__()
        if args.intermediate_size is None:
            intermediate_size = int(args.hidden_size * 8 / 3)
            args.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64) #将计算出的中间层维度向上取整到 64 的整数倍
        
        self.up_proj   = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

        self.dropout = nn.Dropout(args.dropout)
        self.act_fn = ACT2FN[args.hidden_act]

    def forward(self,x):
        return self.dropout(self.down_proj(self.up_proj(x) * self.act_fn(self.gate_proj(x))))

class MoEGate(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        # 每个token选择的专家数量
        self.top_k = config.num_experts_per_tok
        # 总的专家数量
        self.n_routed_experts = config.n_routed_experts
        # 打分函数选择 默认softmax
        self.scoring_func = config.scoring_func
        # 辅助损失的alpha参数
        self.alpha = config.aux_loss_alpha
        # 是否在序列级别上计算辅助损失
        self.seq_aux = config.seq_aux
        # 是否标准话top-k概率
        self.norm_topk_prob = config.norm_topk_prob

        self.gating_dim = config.hidden_size
        # 门控权重：形状 [n_routed_experts, gating_dim]，每个专家对应一个线性投影
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # kaiming初始化
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        # moe只看token值，不关心位置，所以可以合并bsz和seq_len维度
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        # 计算每个token对每个专家的logits：[bsz*seq_len, n_routed_experts]
        # linear映射计算计算出每个token对每个专家的logits
        logits = F.linear(hidden_states, self.weight, None)
        # 对logits做softmax，得到每个token对每个专家的概率（打分）
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1) # [bsz*seq_len, n_routed_experts]
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')
        # 选择Top-K专家：返回权重和索引，维度均为 [bsz*seq_len, top_k]
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        # 归一化Top-K权重：保证选中的专家权重和为1（避免权重分布不均）
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        # 训练阶段且辅助损失系数>0时，计算负载均衡的辅助损失
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            # 恢复batch维度：[bsz, seq_len*top_k]
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            # 序列级别辅助损失
            if self.seq_aux:
                # 序列级辅助损失：按batch维度计算每个专家的负载
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1) # [bsz, seq_len, n_routed_experts]
                # 初始化负载统计张量：
                # ce:[bsz, n_routed_experts],表示每个专家被选中的次数
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                # 统计每个专家被选中的次数（scatter_add_是累加操作）
                # 假设topk_idx_for_aux_loss[2,2],且topk_idx_for_aux_loss[0] = [1,3] 则ce[0][1]和ce[0][3]各加1
                # div_()均衡负载后每个专家的ce值应该是1
                # 大于1说明该专家被过度使用
                ce.scatter_add_(
                    dim=1,
                    index=topk_idx_for_aux_loss,
                    src=torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device),
                ).div_(seq_len * aux_topk / self.n_routed_experts) # seq_len * aux_topk 每个样本总共选中的专家的次数
                # 计算辅助损失：惩罚负载与打分不匹配的情况（鼓励专家负载均匀）
                # scores_for_seq_aux.mean(dim=1)每个样本对每个专家的平均打分 计算后形状[bsz, n_routed_experts]
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            # 批次级别辅助损失
            else:
                #按所有token计算每个专家的负载
                # 转换为one-hot编码：[bsz*seq_len*top_k, n_routed_experts]
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0) # 每个专家被选中的频率：[n_routed_experts]
                Pi = scores_for_aux.mean(0)  # 每个专家的平均打分：[n_routed_experts]
                fi = ce * self.n_routed_experts # 负载归一化（期望为1）
                # 辅助损失核心公式：Pi*fi的和（fi越接近1，损失越小）
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = scores.new_zeros(1).squeeze()

        return topk_idx, topk_weight, aux_loss


class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        # 专家层
        self.experts = nn.ModuleList(
            [FeedForward(config) for _ in range(config.n_routed_experts)]
        )

        # 门控层  
        self.gate = MoEGate(config)
        # 初始化共享专家（所有token都会经过共享专家，提升鲁棒性）
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList(
                [FeedForward(config) for _ in range(config.n_shared_experts)]
            )

    def forward(self, x):
        """
        前向传播：路由token到Top-K专家，计算专家输出的加权和，加上共享专家输出
        Args:
            x: 输入张量，形状 [batch_size, seq_len, hidden_size]
        Returns:
            y: 输出张量，形状 [batch_size, seq_len, hidden_size]
        """
        identity = x 
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        # 使用门控机制选择专家 
        # 1. 门控网络：为每个token选择Top-K专家，得到索引、权重、辅助损失
        topk_idx, topk_weight, aux_loss = self.gate(x)
        # 2. 维度变换：合并batch和seq维度，方便后续计算
        x = x.view(-1, x.shape[-1]) # [bsz*seq_len, hidden_size]
        flat_topk_idx = topk_idx.view(-1) # [bsz*seq_len*top_k]

        # 3. 训练阶段：直接重复token（每个token对应top_k个副本），按专家索引分发计算
        if self.training:
            # 重复token：每个token复制top_k份，形状 [bsz*seq_len*top_k, hidden_size]
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            # 初始化输出张量
            y = torch.empty_like(x, dtype=x.dtype)

            # 遍历每个专家，计算该专家负责的token输出
            for i, expert in enumerate(self.experts):
                # 找到当前专家负责的token索引
                expert_out = expert(x[flat_topk_idx == i])
                if expert_out.shape[0] > 0: 
                    y[flat_topk_idx == i] = expert_out.to(y.dtype)
                else: 
                    # 防止专家无输入时梯度消失（加0*参数和）
                    y[flat_topk_idx == i] = expert_out.to(y.dtype) + 0 * sum(p.sum() for p in expert.parameters())
            # 加权求和：先恢复top_k维度，再按权重加权，最后求和
            # y形状从 [bsz*seq_len*top_k, hidden_size] -> [bsz*seq_len, top_k, hidden_size]
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            # 恢复原始形状：[bsz, seq_len, hidden_size]
            y = y.view(*orig_shape)

        # 4. 推理阶段：优化计算（排序+分块计算），避免重复token，提升效率
        else:

            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)

        # 5. 加上共享专家的输出（所有token都会经过共享专家）
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        """
        推理阶段的MoE计算优化：按专家排序，分块计算，避免重复token，提升效率
        Args:
            x: 输入张量，形状 [bsz*seq_len, hidden_size]
            flat_expert_indices: 展平的专家索引，形状 [bsz*seq_len*top_k]
            flat_expert_weights: 展平的专家权重，形状 [bsz*seq_len*top_k, 1]
        Returns:
            expert_cache: 加权后的输出张量，形状 [bsz*seq_len, hidden_size]
        """
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        # 统计每个专家的token数量（累积和）：[n_routed_experts] cumsum(0)（沿 0 维，一维数组仅 0 维）计算前缀累积和
        # 假设flat_expert_indices.bincount().cpu().numpy() -> [6,9,5,6]
        # cumsum(0) -> [6,15,20,26]
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        # 计算每个索引对应的原始token位置（因为idxs是top_k展平后的索引）
        token_idxs = idxs // self.config.num_experts_per_tok
        # 当tokens_per_expert = [6, 15, 20, 26]，tokens_per_expert.shape[0]即为专家数量（此时为4）
        # 且token_idxs = [3, 7, 19, 21, 24, 25,  4,  5,  6, 10, 11, 12...] 时
        # 意味token_idxs[:6） -> [3, 7, 19, 21, 24, 25]这6个位置属于专家0处理的token（每个token有可能被多个专家处理，这取决于num_experts_per_tok）
        # 接下来9个位置token_idxs[6:15） -> [4,  5,  6, 10, 11, 12...]属于专家1处理的token...依此类推

        # 遍历每个专家，分块计算
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            # 获取当前专家
            expert = self.experts[i]
            # 获取该专家负责的原始token索引
            exp_token_idx = token_idxs[start_idx:end_idx]
            # 提取该专家需要处理的token
            expert_tokens = x[exp_token_idx]
            # 专家前向计算
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            # 乘以对应权重
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            # 累加结果到缓存（按token索引）
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache


class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config:MiniMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        self.self_attn = Attention(config)
        
        self.layer_id = layer_id
        
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps) # 注意力层的输入归一化
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps) # 前馈网络的输入归一化
        
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(
        self,
        hidden_states,
        position_embeddings,
        past_key_value=None,
        use_cache=None,
        attention_mask=None,
    ):
        residual = hidden_states
        hidden_states, present_kv = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask,
        )
        hidden_states += residual

        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))

        return hidden_states, present_kv

class MiniMindModel(nn.Module):
    def __init__(self, config:MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size # 词汇表大小
        self.num_hidden_layers = config.num_hidden_layers #隐藏层数量

        # 1. 词嵌入层：将token ID转换为稠密向量
        # vocab_size: 词汇表大小；hidden_size: 每个token的嵌入维度
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # 2. Dropout层：训练时随机丢弃部分神经元，防止过拟合
        self.dropout = nn.Dropout(config.dropout)

        # 3. 堆叠Transformer块：用ModuleList管理多个MiniMindBlock
        # 为每一层分配唯一的layer_id（0到num_hidden_layers-1），传入配置
        self.layers = nn.ModuleList([MiniMindBlock(i, config) for i in range(self.num_hidden_layers)])

        # 4. 最终归一化层：所有Transformer块输出后的全局RMSNorm
        self.norm = RMSNorm(config.hidden_size, eps = config.rms_norm_eps)

        # 5. 预计算RoPE（旋转位置编码）的余弦/正弦值
        # dim: 单个注意力头的维度（hidden_size / num_attention_heads）
        # end: 最大序列长度（预计算到该长度，避免推理时重复计算）
        # rope_base/rope_scaling: RoPE的基础参数（控制位置编码的缩放）
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        # 将预计算的RoPE值注册为buffer（不参与梯度更新，但随模型移动设备）
        # persistent=False: 保存模型时不保存该buffer（可重新计算，节省空间）
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, 
                input_ids: torch.Tensor | None,
                attention_mask: torch.Tensor | None,
                past_key_values: List[Tuple[torch.Tensor,torch.Tensor]] | None,
                use_cache: bool=False,
                **kwargs):
        """
        Args:
            input_ids: 输入的token ID序列，shape=[batch_size, seq_length]
            attention_mask: 注意力掩码，防止关注padding或未来token
            past_key_values: 历史KV缓存（推理时用，加速生成），list长度=层数，每个元素=(k,v)
            use_cache: 是否启用KV缓存（训练=False，推理=True）
            **kwargs: 兼容其他扩展参数
        Returns:
            hidden_states: 模型最终输出的隐藏状态，shape=[batch_size, seq_length, hidden_size]
            presents: 本轮更新后的KV缓存，供下一轮推理使用
            aux_loss: MoE（混合专家）的辅助损失（普通MLP时为0）
        """
        # 获取输入的批量大小和序列长度
        batch_size, seq_len = input_ids.shape
        if hasattr(past_key_values, 'layers'): 
            past_key_values = None
        # 初始化past_key_values：若为None则创建长度=层数的列表，每个元素为None
        past_key_values = past_key_values or [None] * len(self.layers)

        # 计算当前序列的起始位置（推理时用：比如生成第2个token时，start_pos=1）
        # 若有历史缓存，取第一层缓存的key的第1维长度（即已生成的token数）；否则start_pos=0
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        # 步骤1：词嵌入 + Dropout
        # embed_tokens(input_ids): 将token ID转为向量，shape=[batch_size, seq_length, hidden_size]
        # dropout: 训练时随机丢弃部分值，增强泛化性
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # 步骤2：获取当前序列对应的RoPE位置编码（余弦+正弦）
        # 从预计算的freqs_cos/freqs_sin中截取[start_pos, start_pos+seq_length)区间
        # 比如推理时生成第2个token，截取[1,2)；训练时截取[0, seq_length)
        position_embeddings = (
            self.freqs_cos[start_pos : start_pos + seq_len],
            self.freqs_sin[start_pos : start_pos + seq_len],
        )

        # 步骤3：遍历所有Transformer块，逐层计算
        presents = [] # 保存每一层更新后的KV缓存
        # 同时遍历层索引、层实例、该层的历史KV缓存
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            # 调用单个MiniMindBlock的forward方法
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            presents.append(present) # 保存该层更新后的KV缓存
        
        # 步骤4：所有层计算完成后，做最终的RMSNorm归一化
        hidden_states = self.norm(hidden_states)
        
        aux_loss = sum([i.mlp.aux_loss for i in self.layers if isinstance(i.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())

        return hidden_states, presents, aux_loss
    
class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    """
    MiniMind模型的因果语言建模封装类
    - 继承PreTrainedModel：Hugging Face标准模型基类（提供配置、保存/加载、设备迁移等功能）
    - 继承GenerationMixin：提供generate()等生成方法（实现文本生成逻辑）
    """
    # 绑定配置类，Hugging Face框架要求：通过模型实例可快速获取配置类型
    config_class = MiniMindConfig

    def __init__(self, config:MiniMindConfig = None):
        # 处理默认配置：若未传入config，则创建MiniMindConfig的默认实例
        self.config = config or MiniMindConfig()
        # 调用父类PreTrainedModel的初始化函数（必须传入config）
        super().__init__(config)

        # 1. 实例化基础模型（核心Transformer架构）
        self.model = MiniMindModel(config)

        # 2. 定义语言头（LM Head）：将隐藏状态映射到词汇表维度
        # hidden_size → vocab_size，无偏置（LLM主流设计，减少参数）
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

        # 3. 权重共享：将词嵌入层（embed_tokens）和语言头（lm_head）的权重绑定
        # 核心优化：减少参数数量，提升训练效率和模型效果
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor,torch.Tensor]]] = None,
                use_cache:bool=False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **args):
        """
        因果语言模型前向传播（训练+推理通用）
        Args:
            input_ids: 输入token ID，shape=[batch_size, seq_length]
            attention_mask: 注意力掩码，shape=[batch_size, seq_length]
            labels: 训练标签（与input_ids同shape），用于计算损失
            past_key_values: KV缓存（推理加速）
            use_cache: 是否启用KV缓存
            logits_to_keep: 保留的logits数量/索引（优化推理效率，仅计算部分token的输出）
            **args: 兼容其他扩展参数
        Returns:
            CausalLMOutputWithPast: 包含损失、logits、缓存、隐藏状态的结构化输出
        """
        # 步骤1：调用基础模型MiniMindModel的forward方法，获取核心输出
        # 返回：隐藏状态、更新后的KV缓存、MoE辅助损失
        hidden_states, past_key_values, aux_loss= self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )
        # 步骤2：处理logits的切片（仅保留需要的token，优化计算效率）
        # 情况1：logits_to_keep是整数（如1）→ 取最后N个token（slice(-1, None)）
        # 情况2：logits_to_keep是张量 → 按索引切片（自定义保留的token）
        # 推理时常用：仅计算最后一个token的logits来生成下一个token
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep,int) else logits_to_keep
        # 将隐藏状态映射到词汇表维度（logits），并仅保留指定切片的token
        # hidden_states[:, slice_indices, :] → 切片后的隐藏状态
        # lm_head输出logits：shape=[batch_size, keep_length, vocab_size]
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        # 步骤3：计算训练损失（仅当传入labels时）
        loss = None
        if labels is not None:
            # 移位操作：因果语言建模的核心（预测下一个token）
            # shift_logits：去掉最后一个token（logits[..., :-1, :]）
            # contiguous()：确保张量内存连续，避免后续view()报错
            shift_logits = logits[..., :-1, :].contiguous()
            # shift_labels：去掉第一个token（labels[..., 1:]）
            # 用第i个token的logits预测第i+1个token的标签
            # 例如：input_ids=[1,2,3] → labels=[2,3,4] → shift_logits对应[1,2]的预测，shift_labels对应[2,3]
            shift_labels = labels[..., 1:].contiguous()

            #交叉熵损失
            loss = F.cross_entropy(  
                # 展平logits：[batch_size, seq_len-1, vocab_size] → [batch_size*(seq_len-1), vocab_size]
                shift_logits.view(-1, shift_logits.size(-1)),
                # 展平labels：[batch_size, seq_len-1] → [batch_size*(seq_len-1)]
                shift_labels.view(-1),
                # 忽略标签为-100的位置（通常是padding token，不参与损失计算）
                ignore_index=-100,
            )
        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
        )
        output.aux_loss = aux_loss

        return output


        

        