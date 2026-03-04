import torch

# # 1. torch.div() 逐元素相除

# a = torch.tensor([10,20,30])
# b = torch.tensor([2,5,3])

# c = torch.div(a, b)
# print(c)
# # tensor([5., 4., 10.])

# # 2. torch.mean() 对张量求平均值
# # 一维张量
# x = torch.tensor([1.,2.,3.])
# print(torch.mean(x, dim=0))
# # tensor(2.)
# # 二维张量
# x = torch.tensor([
#     [1.,2.,3.],
#     [4.,5.,6.]
# ]) 
# # dim = 0 对行求平均，列不变
# print(torch.mean(x, dim=0).shape) #[2,3]->[3]
# # tensor([2.5, 3.5, 4.5])
# # dim = 1 对列求平均，行不变
# print(torch.mean(x, dim=1).shape) #[2,3]->[2]
# # tensor([2., 5.])
# # ps dim就是需要消掉的那个维度

# # 3. torch.scatter_add_() 按索引，把值加到目标张量指定位置（原地操作）
# out = torch.zeros(5)
# index = torch.tensor([0,1,0,3])
# src = torch.tensor([1.,2.,3.,4.])

# out.scatter_add_(dim=0, index=index, src=src)

# print(out)
# # tensor([4., 2., 0., 4., 0.])
# """
# - index=0 → 加了 1 和 3 → 4
# - index=1 → 加了 2
# - index=3 → 加了 4
# """

# 4. torch.repeat_interleave()
# 对张量的每个元素进行指定次数重复
# torch.repeat_interleave(input, repeats, dim=None, output_size=None)

# 无 dim 参数，展平后重复
# # 一维张量 
# x = torch.tensor([1, 2, 3])
# # 所有元素重复2次
# y = torch.repeat_interleave(x, repeats=2)

# print("输入：", x)       # tensor([1, 2, 3])
# print("输出：", y)       # tensor([1, 1, 2, 2, 3, 3])
# print("输出形状：", y.shape)  # torch.Size([6])

# # 二维张量（2行2列）
# x = torch.tensor([[1, 2], 
#                   [3, 4]])
# print("原始张量：\n", x)  # 形状: [2, 2]

# # 沿dim=0（行维度）重复，每行元素重复2次
# y1 = torch.repeat_interleave(x, repeats=2, dim=0)
# print("\n沿dim=0重复：\n", y1)
# print("形状：", y1.shape)  # [4, 2] → 2行×2次=4行

# # 沿dim=1（列维度）重复，每列元素重复2次
# y2 = torch.repeat_interleave(x, repeats=2, dim=1)
# print("\n沿dim=1重复：\n", y2)
# print("形状：", y2.shape)  # [2, 4] → 2列×2次=4列
# """
# 原始张量：
#  tensor([[1, 2],
#         [3, 4]])

# 沿dim=0重复：
#  tensor([[1, 2],
#         [1, 2],
#         [3, 4],
#         [3, 4]])
# 形状： torch.Size([4, 2])

# 沿dim=1重复：
#  tensor([[1, 1, 2, 2],
#         [3, 3, 4, 4]])
# 形状： torch.Size([2, 4])
# """
# # 自定义每个元素的重复次数
# x = torch.tensor([1, 2, 3])
# # 第一个元素重复1次，第二个重复2次，第三个重复3次
# repeats = torch.tensor([1, 2, 3])
# y = torch.repeat_interleave(x, repeats=repeats)

# print("输入：", x)         # tensor([1, 2, 3])
# print("重复次数：", repeats)  # tensor([1, 2, 3])
# print("输出：", y)         # tensor([1, 2, 2, 3, 3, 3])

# 5. torch.argsort()
# 返回排序后元素在源数组中的索引
# 排序但不破坏原数据

# x = torch.tensor([30,10,20])
# idx = torch.argsort(x)

# print(idx)# tensor([1, 2, 0])
# print(x[idx])# tensor([10, 20, 30])

# 6. torch.bincount() 统计非负整数出现的次数
x = torch.tensor([0,1,1,3,1])
count = torch.bincount(x)

print(count)
# tensor([1, 3, 0, 1]) 0出现1次 1出现3次 2出现0次 3出现1次 

