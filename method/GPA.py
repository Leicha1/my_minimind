# ruff: noqa: F401
import torch
import torch.nn as nn


# # 1. nn.Dropout()
# dropout = nn.Dropout(p=0.5)  # 设置丢弃概率为0.5
# input_tensor = torch.tensor([1.,2.,3.])  
# output_tensor = dropout(input_tensor)  # 应用Dropout

# print(output_tensor)

# # 2.nn.Linear()
# #全连接层,作用是把输入的特征映射到新的特征空间
# #torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
# #输入张量的最后一维必须等于 in_features

# linear = nn.Linear(in_features=5, out_features=3)
# input_tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])  # 必须是浮点型！
# # 应用线性层（自动完成 y = xA^T + b 计算）
# output_tensor = linear(input_tensor)
# print("输入形状：", input_tensor.shape)  # torch.Size([1, 5])
# print("输出形状：", output_tensor.shape)  # torch.Size([1, 3])
# print("输出值：\n", output_tensor)

# linear = nn.Linear(10, 2)

# # 构造输入：64个样本（批量大小），每个样本10个特征（形状 [64, 10]）
# batch_input = torch.randn(64, 10)  # 随机生成浮点型张量
# # 线性变换
# batch_output = linear(batch_input)
# print("批量输入形状：", batch_input.shape)  # torch.Size([64, 10])
# print("批量输出形状：", batch_output.shape)  # torch.Size([64, 2])

# # 3. torch.view()/torch.reshape()
# # 都用于改变张量维度，但view()只适用于连续张量，reshape()则兼容
# # view返回的是原张量的「视图」（共享内存），修改视图会同步修改原张量。
# # reshape在张量连续时同view一致，张量非连续时自动创建拷贝（不共享内存）

# x = torch.arange(12)
# print(x.shape)
# x_view = x.view(3,4)
# print(x_view.shape) # torch.Size([12])
# print(x_view)
# # tensor([[ 0,  1,  2,  3],
# #         [ 4,  5,  6,  7],
# #         [ 8,  9, 10, 11]])
# x_view[0,0] = 999.
# print(x) #tensor([999,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11])

# x_reshape = x.reshape(2,6) #torch.Size([2, 6])
# print(x_reshape.shape)
# print(x_reshape)

# # 4.torch.transpose()
# # 专门用于交换张量的两个维度（转置），返回的是原张量的视图（共享内存），但会让张量变成「非连续内存」
# # 矩阵转置（交换0和1维度）
# x = torch.arange(12).view(3,4)  # [3,4]
# x_trans = x.transpose(0, 1)     # 交换行和列，形状 [4,3]
# print("x_trans 形状：", x_trans.shape)  # torch.Size([4,3])

# # 共享内存（修改转置张量影响原张量）
# x_trans[0,0] = 888
# print("原张量x：\n", x)  # 第一行第一列变成888

# # 高维张量转置（比如图片张量 [B,C,H,W] → 交换通道和高度）
# img = torch.randn(2, 3, 28, 28)  # 2张图，3通道，28×28
# img_trans = img.transpose(1,2)   # 交换通道(C=1)和高度(H=2)，形状 [2,28,3,28]
# print("img_trans 形状：", img_trans.shape) #torch.Size([2, 28, 3, 28])

# 5.torch.triu()/tril() 生成上三角矩阵/下三角矩阵
#torch.triu/tril(input, diagonal=0, *, out=None) diagonal控制保留的对角线位置
# 创建5×5全1矩阵
x = torch.ones(3, 3)
# 生成上三角矩阵（保留主对角线及以上）
triu_x = torch.triu(x)
#  tensor([[1., 1., 1.],
#          [0., 1., 1.],
#          [0., 0., 1.]])
triu_x2 = torch.triu(x,1) #保留主对角线上一条对角线及以上
#  tensor([[0., 1., 1.],
#          [0., 0., 1.],
#          [0., 0., 0.]])
print("原矩阵：\n", x)
print("\n上三角矩阵（diagonal=0）：\n", triu_x)
print("\n上三角矩阵（diagonal=1）：\n", triu_x2)