# ruff:noqa: F401
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')

def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    """
    单轮epoch的训练循环
    Args:
        epoch: 当前训练的epoch数（从0开始）
        loader: PyTorch DataLoader，加载训练数据（input_ids, labels）
        iters: 当前epoch的总迭代步数（len(loader)）
        start_step: 起始步数（断点续训时用，默认0）
        wandb: Weights & Biases日志实例（可选，用于可视化训练过程）
    """
    # 记录当前epoch的开始时间（用于计算耗时）
    start_time = time.time()

    # 遍历DataLoader，enumerate的start参数设置起始步数（断点续训）
    # step：当前迭代步数（从start_step+1开始）；(input_ids, labels)：批量数据
    for step, (input_ids, labels, attention_mask) in enumerate(loader, start=start_step+1):
        # ========== 步骤1：数据设备迁移 ==========
        # 将输入数据移到指定设备（GPU/CPU），args.device通常是"cuda"或"cpu"
        input_ids = input_ids.to(args.device)
        labels =labels.to(args.device)
        attention_mask = attention_mask.to(args.device)  # ！修正：接收并转移 attention_mask

        # ========== 步骤2：动态调整学习率 ==========
        # 计算当前全局步数：epoch*总步数 + 当前step（用于学习率调度，余弦退火）
        global_step = epoch * iters + step
        # 总训练步数：总epoch数 * 每epoch步数
        total_steps = args.epochs * iters
        # 获取当前步数对应的学习率
        lr = get_lr(global_step, total_steps, args.learning_rate)
        # 将计算出的学习率应用到优化器的所有参数组
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # ========== 步骤3：前向传播（混合精度训练） ==========
        # autocast_ctx：混合精度上下文（torch.cuda.amp），自动混合float16/float32，提升训练速度、节省显存\
        with autocast_ctx:
            # 模型前向传播：输入input_ids和labels，返回包含loss的结果对象
            res = model(input_ids, labels=labels, attention_mask=attention_mask)  # ！修正：直接传入labels和attention_mask，由模型内部计算loss
            # 总损失 = 主损失（logits_loss） + MoE辅助损失（aux_loss）
            # 主损失：因果语言建模的交叉熵损失；aux_loss：MoE的负载平衡损失（普通模型为0）
            loss = res.loss + res.aux_loss
            # 梯度累积：将损失除以累积步数（避免累积后梯度放大）
            # 比如accumulation_steps=4，每4步更新一次参数，每步损失除以4
            loss = loss / args.accumulation_steps

        # ========== 步骤4：反向传播（混合精度） ==========
        # scaler：梯度缩放器（torch.cuda.amp.GradScaler），解决float16梯度下溢问题
        # 缩放损失并反向传播，计算梯度
        scaler.scale(loss).backward()

        # ========== 步骤5：梯度累积达到阈值，更新参数 ==========
        if (step + 1) % args.accumulation_steps == 0:
            # 反缩放梯度（为梯度裁剪做准备）
            scaler.unscale_(optimizer)
            # 梯度裁剪：限制梯度的最大范数（args.grad_clip，如1.0），防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # 应用缩放后的梯度，更新模型参数
            scaler.step(optimizer)
            # 更新scaler的缩放因子（自适应调整）
            scaler.update()

            # 清空梯度：set_to_none=True比zero_grad()更高效（释放显存）
            optimizer.zero_grad(set_to_none=True)

        # ========== 步骤6：日志记录（按间隔/最后一步） ==========
        if step % args.log_interval == 0 or step == iters - 1:
            # 计算已花费的时间（秒）
            spend_time = time.time() - start_time
            # 恢复真实损失值（乘以累积步数，因为之前除以了accumulation_steps）
            current_loss = loss.item() * args.accumulation_steps
            # 获取辅助损失值（MoE时非0，普通模型为0）
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            # 计算主损失（总损失 - 辅助损失）
            current_logits_loss = current_loss - current_aux_loss
            # 获取当前学习率（取最后一个参数组的lr，通常所有参数组lr相同）
            current_lr = optimizer.param_groups[-1]['lr']
            # 计算剩余时间（分钟）：已用时间/已走步数 * 总步数 - 已用时间 → 转分钟
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            # 打印训练日志（自定义Logger，如打印到控制台/文件）
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            # 如果启用wandb，记录日志到可视化平台
            if wandb: 
                wandb.log({
                    "loss": current_loss, 
                    "logits_loss": current_logits_loss, 
                    "aux_loss": current_aux_loss, 
                    "learning_rate": current_lr, 
                    "epoch_time": eta_min
                })

        # ========== 步骤7：模型保存（按间隔/最后一步，仅主进程执行） ==========
        # is_main_process()：分布式训练时，仅主进程（rank=0）保存模型，避免冲突
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            # 切换模型到评估模式（防止Dropout/BatchNorm等层影响保存）
            model.eval()
            
            # 生成模型文件名后缀：MoE模型加'_moe'，普通模型为空
            moe_suffix = '_moe' if lm_config.use_moe else ''
            # 模型保存路径：保存目录 + 权重名 + 隐藏层大小 + 后缀 + .pth
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            
            # 处理分布式模型：如果是DDP包装的模型，取原始模型（module）
            raw_model = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
            # 兼容Hugging Face的模型包装（_orig_mod是原始模型）
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            
            # 获取模型的状态字典（参数）
            state_dict = raw_model.state_dict()
            # 保存模型：将参数转为半精度（half）并移到CPU，节省磁盘空间
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            
            # 保存完整的检查点（包含模型、优化器、scaler、epoch、step等，断点续训用）
            lm_checkpoint(
                lm_config, 
                weight=args.save_weight, 
                model=model, 
                optimizer=optimizer, 
                scaler=scaler, 
                epoch=epoch, 
                step=step, 
                wandb=wandb, 
                save_dir='../checkpoints'
            )
            
            # 切换回训练模式
            model.train()
            # 释放状态字典显存
            del state_dict

        # ========== 步骤8：清理显存（关键，防止OOM） ==========
        del input_ids, labels, res, loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    # ========== 基础训练参数 ==========
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=6, help="训练轮数（建议1轮zero或2-6轮充分训练）")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")

    # ========== 硬件和性能参数 ==========
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")

    # ========== 训练策略参数 ==========
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")

    # ========== 模型架构参数 ==========
    parser.add_argument('--hidden_size', default=640, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=1, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")

    # ========== 数据和恢复参数 ==========
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_hq.jsonl", help="预训练数据路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")

    # ========== 实验跟踪参数 ==========
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    """
    📚 分布式训练初始化知识点：
    - local_rank: 当前进程在本机上的GPU编号
    - 随机种子: 确保不同进程有不同但可复现的随机序列
    - 这样既保证了随机性，又保证了可复现性
    """
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"  # 分布式训练时使用对应的GPU

    # 📚 随机种子设置知识点
    # 不同进程使用不同的种子，避免数据采样完全相同
    # 42是基础种子，每个进程加上自己的rank保证不同
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查点 ==========
    """
    📚 模型配置和检查点管理：
    - 创建保存目录
    - 构建模型配置对象
    - 尝试加载断点续训数据
    """
    os.makedirs(args.save_dir, exist_ok=True)  # 确保保存目录存在

    # 创建MiniMind模型配置
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )

    # 📚 断点续训知识点
    # 如果开启了断点续训，尝试加载之前的训练状态
    ckp_data = (
        lm_checkpoint(
            lm_config, weight=args.save_weight, save_dir="../checkpoints"
        )  # ！修正：原"checkpoints"缺少../前缀
        if args.from_resume == 1
        else None
    )
    # ========== 3. 设置混合精度 ==========
    """
    📚 混合精度训练知识点：
    - bfloat16: Google开发，数值范围大，更稳定
    - float16: 标准半精度，节省内存但可能溢出
    - autocast: 自动选择精度，关键运算用float32
    """
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    # 📚 上下文管理器知识点
    # CPU不支持autocast，使用nullcontext作为空操作
    autocast_ctx = (
        nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    )

    # ========== 4. 配置WandB实验跟踪 ==========
    """
    📚 实验跟踪系统知识点：
    - WandB: 实验管理平台，记录训练过程
    - SwanLab: 国产替代方案
    - 支持断点续训时恢复到同一个实验
    """
    wandb = None
    if args.use_wandb and is_main_process():
        # 使用SwanLab作为WandB的替代
        # import swanlab as wandb

        # 📚 实验恢复知识点
        # 如果有检查点数据，获取之前的wandb_id来恢复实验
        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        resume = "must" if wandb_id else None  # 必须恢复到指定实验

        # 构建实验名称，包含关键超参数
        wandb_run_name = f"MokioMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(
            project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume
        )

    # ========== 5. 定义模型、数据、优化器 ==========
    """
    📚 训练组件初始化：
    - 模型: 根据配置创建MiniMind模型
    - 数据集: 加载预训练数据
    - 采样器: 分布式训练的数据分配
    - 优化器: AdamW优化器
    - 缩放器: 混合精度训练的梯度缩放
    """
    # 初始化模型和分词器
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)

    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)

    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None

    scaler = torch.amp.GradScaler(enabled=(args.dtype == "float16"))

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    start_epoch, start_step = 0, 0
    if ckp_data:
        # 恢复模型参数
        model.load_state_dict(ckp_data["model"])
        # 恢复优化器状态（动量、方差估计等）
        optimizer.load_state_dict(ckp_data["optimizer"])
        # 恢复梯度缩放器状态
        scaler.load_state_dict(ckp_data["scaler"])
        # 恢复训练进度
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step", 0)

    if dist.is_initialized():
        # 📚 RoPE位置编码特殊处理
        # freqs_cos, freqs_sin是位置编码缓存，不需要梯度同步
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    for epoch in range(start_epoch, args.epochs):
        # 📚 分布式采样器epoch设置
        # 每个epoch设置不同的随机种子，确保数据顺序随机化
        if train_sampler:
            train_sampler.set_epoch(epoch)

        # 📚 断点续训逻辑
        if epoch == start_epoch and start_step > 0:  # 第一个epoch且存在检查点
            # 使用跳批采样器，跳过已训练的数据
            batch_sampler = SkipBatchSampler(
                train_sampler or range(len(train_ds)), args.batch_size, start_step
            )
            loader = DataLoader(
                train_ds,
                batch_sampler=batch_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            Logger(
                f"Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始"
            )
            train_epoch(epoch, loader, len(loader) + start_step, start_step, wandb)
        else:  # 默认从头开始
            loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            train_epoch(epoch, loader, len(loader), 0, wandb)

