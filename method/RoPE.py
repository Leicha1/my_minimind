import torch


#1.torch.where()
# x = torch.tensor([1,2,3,4,5])
# y = torch.tensor([10,20,30,40,50])

# condition = x > 3

# result = torch.where(condition, x, y) # 保留第一个张量（x）中满足条件condition的位置，剩下位置由y张量对应位置填充 
# print(result) #output tensor([10, 20, 30, 4, 5])

#2.torch.arange()

# print(torch.arange(0,10,2)) # [0,10) 步长为2的序列
# print(torch.arange(5,0,-1)) # [5,0) 步长为-1的序列
# # tensor([0, 2, 4, 6, 8])
# tensor([5, 4, 3, 2, 1])

#3.torch.outer() 只接收一维张量，返回两个一维张量的外积（二维矩阵）
# v1 = torch.tensor([1,2,3])
# v2 = torch.tensor([4,5,6])
# print(torch.outer(v1,v2))
'''
tensor([[ 4,  5,  6],
        [ 8, 10, 12],
        [12, 15, 18]])
'''

#4.torch.cat() 把多个形状兼容的张量，沿着指定维度拼接成一个新张量
'''
torch.cat(tensors, dim=0, out=None)
tensors：要拼接的张量列表（比如[tensor1, tensor2]）；
dim：指定拼接的维度（核心参数），默认是 0；
out：可选，指定输出张量（一般不用）。
ps:除了拼接维度，其他维度的形状必须完全一致
'''
t1 = torch.tensor([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]]) # shape[2,2,3]
t2 = torch.tensor([[[13,14,15],[16,17,18]],[[19,20,21],[22,23,24]]])
print(t1.shape)  #  torch.Size([2, 2, 3])
result = torch.cat((t1,t2),dim=-1)
print(result.shape) # torch.Size([4, 2, 3])
print(result)

#5.troch.unsqueeze() 加一维
# x.unsqueeze(dim) dim:加维的维度
t1 = torch.tensor([1,2,3]) #一维张量
t2 = t1.unsqueeze(0)
print(t1.shape) #torch.Size([3])
print(t2.shape) #torch.Size([1, 3])
print(t2) #tensor([[1, 2, 3]])

t3 = t1.unsqueeze(1)
print(t3.shape) #torch.Size([3, 1])
print(t3) #tensor([[1],
          #        [2],
          #        [3]])