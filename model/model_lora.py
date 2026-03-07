import torch
from torch import nn

# 定义LoRA网络结构
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank #LoRA的秩，控制低秩矩阵的大小
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)

        # 矩阵A高斯初始化
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # 矩阵B全0初始化
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x))
    
def apply_lora(model, rank = 8):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            lora = LoRA(module.weight.shape[0], module.weight.shape[1],rank).to(model.device)
            setattr(module, "lora", lora)
            original_forward = module.forward

            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)
            
            module.forward = forward_with_lora
            
def load_lora(model,path):
    # 1. 加载LoRA权重文件，自动映射到模型所在设备（CPU/GPU）
    state_dict = torch.load(path, map_location=model.device)
    # 2. 清理权重键名：移除可能存在的'module.'前缀（多卡训练时会自动加这个前缀）
    state_dict = {(k[7:] if k.startswith('module.') else k) : v for k, v in state_dict.items()}
    # 3. 遍历模型的所有子模块，寻找包含lora属性的模块
    for name, module in model.named_modules():
        if hasattr(module,'lora'):
            # 4. 筛选出当前lora模块对应的权重（只保留以"模块名.lora."开头的键）
            lora_state = {k.replace(f'{name}.lora.', "") : v for k, v in state_dict.items() if f'{name}.lora.' in k}
            # 5. 将筛选后的权重加载到当前模块的lora组件中
            module.lora.load_state_dict(lora_state)

def save_lora(model, path):
    # 1. 处理被封装的模型：如果模型被_original_mod封装（如DeepSpeed/PEFT封装），取原始模型
    raw_model = getattr(model,"_orig_mod", model)
    state_dict = {}
    # 2. 遍历原始模型的所有子模块
    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'):
            # 3. 清理模块名：移除可能的'module.'前缀
            clean_name = name[7:] if name.startswith("module.") else name
            # 4. 构建LoRA权重键名：格式为"清理后的模块名.lora.权重名"
            lora_state = {f'{clean_name}.lora.{k}' : v for k, v in module.lora.state_dict().items()}
            # 5. 将当前模块的LoRA权重合并到总字典中
            state_dict.update(lora_state)
    torch.save(state_dict,path)


