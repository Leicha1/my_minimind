
"""
MiniMind LoRA (Low-Rank Adaptation) 微调脚本

📚 LoRA 核心知识点：
- 什么是LoRA：一种参数高效微调方法，只训练少量新增参数
- 原理：在预训练模型的权重矩阵旁边添加低秩分解矩阵 ΔW = BA
  - 原始权重 W 保持冻结（requires_grad=False）
  - 新增两个小矩阵 A(dxr) 和 B(rxd)，其中 r<<d（秩远小于维度）
  - 前向计算：output = Wx + BAx
- 优势对比：
  - Full SFT：更新所有参数，效果好但需要大显存和长时间
  - LoRA：只更新1-5%的参数，显存需求小，训练快，适合资源受限场景
  - 多任务切换：可以保存多组LoRA权重，快速切换不同任务能力

📚 适用场景：
- 个性化定制：医疗、法律、金融等垂直领域适配
- 快速实验：尝试不同数据/超参时，LoRA训练速度快
- 资源受限：单卡或小显存环境
"""
import os
import sys
# 📚 Python模块系统
# __package__: 显式声明当前模块所属的包
# sys.path.append: 将项目根目录加入模块搜索路径，使得可以导入project内的模块
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse  # 命令行参数解析
import time  # 时间统计
import warnings  # 警告控制
import torch  # PyTorch深度学习框架
import torch.distributed as dist  # 分布式训练支持
from contextlib import nullcontext  # 上下文管理器（无操作占位符）
from torch import optim  # 优化器
from torch.nn.parallel import DistributedDataParallel  # 分布式数据并行
from torch.utils.data import DataLoader, DistributedSampler  # 数据加载

# MokioMind相关组件
from model.model_minimind import MiniMindConfig  # 模型配置
from dataset.lm_dataset import SFTDataset  # 监督微调数据集
from model.model_lora import save_lora, apply_lora  # LoRA权重保存和应用
from trainer.trainer_utils import (  # 训练工具函数
    get_lr,
    Logger,
    is_main_process,
    lm_checkpoint,
    init_distributed_mode,
    setup_seed,
    init_model,
    SkipBatchSampler,
)
warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, lora_params, start_step=0, wandb=None):
    """
    执行单个LoRA训练轮次
    📚 LoRA训练的特殊之处：
    1. 只有LoRA参数参与梯度计算和更新
    2. 原始模型权重保持冻结，节省显存和计算
    3. 训练流程与Full SFT相同，但参数量小得多
    Args:
        epoch: 当前训练轮次
        loader: 数据加载器
        iters: 总迭代次数
        lora_params: LoRA参数列表（只有这些参数会被更新）
        start_step: 起始步数（用于断点续训）
        wandb: 实验跟踪工具
    """
    start_time = time.time()    
    for step, (input_ids, labels, attention_mask) in enumerate(loader, start=start_step + 1):
        
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        attention_mask = attention_mask.to(args.device)  # ！修正：接收并转移attention_mask

        # 📚 学习率调度：使用余弦退火+预热策略
        # 从初始学习率逐渐降低到接近0
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # 📚 混合精度训练上下文
        # autocast_ctx: 自动混合精度，关键运算用float32，其他用float16/bfloat16
        # 可以加速训练并节省显存，同时保持数值稳定性
        with autocast_ctx:
            res = model(input_ids, labels=labels,attention_mask=attention_mask)
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps
        # 📚 混合精度反向传播
        # scaler.scale(loss): 放大损失值，防止float16下的梯度下溢
        # .backward(): 计算梯度，填充到各参数的.grad属性
        scaler.scale(loss).backward()
        # 📚 梯度累积和参数更新
        # 每accumulation_steps步才真正更新一次参数
        if (step + 1) % args.accumulation_steps == 0:
            # 📚 梯度反缩放
            # scaler.unscale_(optimizer): 将放大的梯度恢复到真实值
            # 必须在梯度裁剪之前调用
            scaler.unscale_(optimizer)

            # 📚 梯度裁剪
            # clip_grad_norm_: 将梯度的L2范数限制在指定阈值内
            # 防止梯度爆炸，稳定训练过程
            # 注意：这里只裁剪lora_params，因为其他参数已被冻结
            torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
            # 📚 优化器步进
            # scaler.step(optimizer): 执行参数更新 param = param - lr * grad
            # scaler.update(): 更新scaler的缩放因子，用于下一次迭代
            scaler.step(optimizer)
            scaler.update()
            # 📚 梯度清零
            # set_to_none=True: 将梯度设为None而不是0
            # 优点：节省内存，性能更好
            optimizer.zero_grad(set_to_none=True)

        # 📚 训练日志记录
        # 每log_interval步或最后一步打印一次日志
        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            # 📚 .item()方法
            # 将单元素张量转换为Python标量
            # 必须恢复梯度累积的缩放：乘以accumulation_steps
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']

            # 📚 ETA计算（预计剩余时间）
            # (已用时间 / 已完成步数) * 总步数 = 预计总时间
            # 预计总时间 - 已用时间 = 预计剩余时间
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: 
                wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            lora_save_path = f'{args.save_dir}/{args.lora_name}_{lm_config.hidden_size}.pth'
            # LoRA只保存LoRA权重
            save_lora(model, lora_save_path)
            lm_checkpoint(lm_config, weight=args.lora_name, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()

        del input_ids, labels, res, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind LoRA Fine-tuning")
    parser.add_argument("--save_dir", type=str, default="../out/lora", help="模型保存目录")
    parser.add_argument("--lora_name", type=str, default="lora_identity", help="LoRA权重名称(如lora_identity/lora_medical等)")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=10, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=640, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=1, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/lora_identity.jsonl", help="LoRA训练数据路径")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练，默认full_sft")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-LoRA", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): 
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.lora_name, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-LoRA-{args.lora_name}-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、应用LoRA、冻结非LoRA参数 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    apply_lora(model)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    lora_params_count = sum(p.numel() for name, p in model.named_parameters() if 'lora' in name)
    Logger(f"LLM 总参数量: {total_params / 1e6:.3f} M")
    Logger(f"LoRA 参数量: {lora_params_count / 1e6:.3f} M")
    Logger(f"LoRA 参数占比: {lora_params_count / total_params * 100:.2f}%")
    
    # 冻结非LoRA参数，收集LoRA参数
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
            lora_params.append(param)
        else:
            param.requires_grad = False
    
    # ========== 6. 定义数据和优化器 ==========
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(lora_params, lr=args.learning_rate)
    
    # ========== 7. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'], strict=False)
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 8. DDP包模型 ==========
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 9. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch) 
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, lora_params, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), lora_params, 0, wandb)
    
    # ========== 10. 清理分布进程 ==========
    if dist.is_initialized(): 
        dist.destroy_process_group()