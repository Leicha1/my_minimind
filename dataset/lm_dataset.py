# ruff:noqa: F401

from torch.utils.data import Dataset
import torch
import os
import random
from datasets import load_dataset

# 关键环境变量设置：禁用tokenizers的并行化，避免多线程冲突（解决警告/报错）
os.environ["TOKENIZERS_PARALLELTSM"] = "false"

def pre_processing_chat(conversations, add_system_ratio=0.2):
    """
    对话前处理：以一定概率随机插入 system 消息。
    特点：
    - 只有当首条消息不是 system 角色时才可能插入。
    - add_system_ratio 控制插入概率（默认 20%），引入随机性可提升模型
      对有/无 system prompt 两种情况的泛化能力。
    - system 内容从预定义的中英文 prompt 池中随机抽取，覆盖不同表达风格。
    """
    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model.",
    ]
    if conversations and conversations[0].get("role") != "system":
        if random.random() < add_system_ratio:
            return [
                {"role": "system", "content": random.choice(SYSTEM_PROMPTS)}
            ] + conversations
    return conversations


def post_processing_chat(prompt_content, empty_think_ratio=0.05):
    """
    对话后处理：清理模板渲染后多余的空 <think> 块。
    特点：
    - 针对带 CoT（chain-of-thought）格式的模型，apply_chat_template 有时会
      渲染出 "<think>\n\n</think>\n\n" 这样的空思考块占位符。
    - 大部分情况下（概率 1 - empty_think_ratio = 95%）直接删除该空块，
      防止模型学到"无意义思考"的坏习惯。
    - 保留少量空思考块（empty_think_ratio = 5%），让模型也能处理该边界情况。
    """
    if (
        "<think>\n\n</think>\n\n" in prompt_content
        and random.random() > empty_think_ratio
    ):
        prompt_content = prompt_content.replace("<think>\n\n</think>\n\n", "")
    return prompt_content

# ──────────────────────────────────────────────────────────────────────────────
# 1. PretrainDataset —— 自回归预训练数据集
# ──────────────────────────────────────────────────────────────────────────────
# 训练目标：Next-Token Prediction（下一个 token 预测）
# 数据格式：{"text": "一段原始文本"}
# 训练特点：
#   - 模型对整段文本的每个位置都进行预测，没有"只学回复"的区分。
#   - 使用 BOS/EOS 标记文本边界，让模型学会文本的起止。
#   - PAD token 对应的 label 置 -100，不参与 loss 计算，节省无效梯度。
#   - labels 直接 clone 自 input_ids（即 X 和 Y 错位一格：Y[t] = X[t+1]）。
# ──────────────────────────────────────────────────────────────────────────────
class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 加载JSON格式的数据集：
        # - 'json'：指定数据集格式
        # - data_files=data_path：指定数据文件路径
        # - split='train'：加载训练集（datasets库的标准写法）
        # 加载后self.samples是一个Dataset对象，可通过索引访问样本
        self.samples = load_dataset("json", data_files = data_path, split = "train")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        """
        根据索引获取单个样本
        Args:
            index: 样本索引（0到len(self.samples)-1）
        Returns:
            input_ids: 处理后的token ID序列，shape=[max_length]，torch.long类型
            labels: 标签序列（与input_ids同shape），padding位置设为-100，torch.long类型
        """
        sample = self.samples[index]
        # 1. 获取原始文本样本：按索引取样本，提取'text'字段（JSON中必须有该字段）
        text = str(sample['text'])

        # 2. 文本分词：
        # - add_special_tokens=False：暂不添加BOS/EOS（后续手动加，更灵活）
        # - max_length=self.max_length - 2：预留2个位置给BOS/EOS
        # - truncation=True：超过max_length-2的文本截断
        # - .input_ids：只取分词后的token ID列表
        tokens = self.tokenizer(
            text,
            add_special_tokens=False,
            max_length=self.max_length - 2,
            truncation=True
        ).input_ids

        # 3. 添加特殊token：BOS（开头） + 正文tokens + EOS（结尾）
        # BOS：Begin of Sequence（序列开始），EOS：End of Sequence（序列结束）
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]

        # 4. 填充（padding）到最大长度：
        # 计算需要填充的pad_token数量：max_length - 当前tokens长度
        # pad_token_id：填充token的ID（通常为0）
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))

        # 5. 转换为PyTorch张量（long类型，符合LLM输入要求）
        input_ids = torch.tensor(input_ids, dtype = torch.long)

        # 6. 生成标签序列（因果语言建模）：
        # - 先克隆input_ids（labels初始值和input_ids完全一致）
        # - 将padding位置的标签设为-100（损失计算时会忽略这些位置）
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        # ！修正：返回 attention_mask，使 attention 层能屏蔽 padding token
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        return input_ids, labels, attention_mask


# ──────────────────────────────────────────────────────────────────────────────
# 2. SFTDataset —— 有监督微调（Supervised Fine-Tuning）数据集
# ──────────────────────────────────────────────────────────────────────────────
# 训练目标：让模型学会"只预测 assistant 回复"，忽略 user/system 输入
# 数据格式：{"conversations": [{"role": "user"/"assistant"/"system", "content": "..."}]}
# 训练特点：
#   - 通过 generate_labels 扫描 bos_id（assistant 回复起始标记）定位每段回复，
#     仅将 assistant 回复的 token 位置设为有效 label，其余全部为 -100。
#   - 这样做的意义：让 loss 只反映模型对"正确回答"的拟合，不浪费梯度在
#     用户输入的复现上（用户输入只作为 context，不是预测目标）。
#   - 支持 function calling：若 system 消息携带 "functions" 字段，
#     会透传给 apply_chat_template，生成带工具描述的提示词。
#   - 与 PretrainDataset 的关键区别：标签是"稀疏"的，只有 assistant 部分非 -100。
# ──────────────────────────────────────────────────────────────────────────────
#修正
class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.samples = load_dataset("json", data_files=jsonl_path, split="train")
        # 预计算 bos_id：对应 "assistant\n" 前缀的 token id（模型生成助手回复的起始标识）
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        # 预计算 eos_id：对应 eos_token + 换行的 token id（回复结束标识）
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)
    
    def create_chat_prompt(self, conversations):
        """
        将多轮对话转换为模型输入的字符串。
        特点：
        - 复制原始 conversations，防止修改原始数据。
        - 检测 system 消息中是否携带 functions 字段（function calling 场景），
          若有则透传给 apply_chat_template，生成标准 tool-use 格式的提示词。
        - add_generation_prompt=False：不在末尾追加"请模型续写"的 prompt，
          因为训练时需要完整的 input+output 序列，而非开放续写。
        """
        messages = conversations.copy()
        # 提取工具调用相关信息（如果第一轮是 system 角色且包含 functions）
        tools = (
            conversations[0]["functions"]
            if (
                conversations
                and conversations[0]["role"] == "system"
                and conversations[0].get("functions")
            )
            else None
        )

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )
    
    def generate_labels(self,input_ids):
        """
        生成 SFT 训练所需的稀疏标签序列。

        算法逻辑（滑动窗口扫描）：
        1. 初始化全 -100 的 labels，默认所有位置不计算 loss。
        2. 逐位扫描 input_ids，检测是否匹配 bos_id（assistant 回复起始）。
        3. 匹配到 bos_id 后，向后扫描直到找到 eos_id（回复结束）。
        4. 将 [start, end+len(eos_id)) 区间内的 label 设为对应的 input_ids 值，
           即这段 assistant 回复参与 loss 计算。
        5. EOS token 本身也计入 label，让模型学会何时停止生成。
        6. 跳过已处理区间，继续扫描下一段 assistant 回复（支持多轮对话）。
        """
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i : i + len(self.bos_id)] == self.bos_id:
                # 跳过 bos_id 本身，从 assistant 实际内容开始
                start = i + len(self.bos_id)
                end = start
                # 向后扫描，找到 eos_id 的位置
                while end < len(input_ids):
                    if input_ids[end:end+len(self.eos_id)] == self.eos_id:
                        break
                    end+=1
                # 将 assistant 回复（含 EOS）区间的 label 设为真实 token id
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                # 跳过已处理的区间，继续查找下一个助手回复
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i+=1
        return labels
    
    def __getitem__(self, index): 
        sample = self.samples[index] # 获取单个样本
        # 预处理对话数据
        conversations = pre_processing_chat(sample["conversations"])
        # 构建格式化的聊天提示文本
        prompt = self.create_chat_prompt(conversations)
        # 后处理提示文本
        prompt = post_processing_chat(prompt)
        # 对提示文本进行token化,截断到 max_length
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        # token长度不足时补充pad_token
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        # 生成对应的labels
        labels = self.generate_labels(input_ids)

        # ！修正：返回 attention_mask，使 attention 层能屏蔽 padding token 
        attention_mask = (torch.tensor(input_ids, dtype=torch.long) != self.tokenizer.pad_token_id).long()
        # 转化为pytorch张量
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long),attention_mask



        


