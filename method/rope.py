import torch
# -- conditional selection --
# x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
# y = torch.tensor([10, 20, 30, 40, 50])

# condition = x > 3.0

# result = torch.where(condition, x, y) # if condition is true, take x, else take y

# print(result) # tensor([10., 20., 30.,  4.,  5.])


# -- element-wise operations --
# t = torch.arange(0, 10, 2)
# print(t) # tensor([0, 2, 4, 6, 8])

# t2 = torch.arange(5, 0, -1)
# print(t2) # tensor([5, 4, 3, 2, 1])


# -- matrix multiplication --
# v1 = torch.tensor([1.0, 2.0, 3.0])
# v2 = torch.tensor([4.0, 5.0, 6.0])

# result = torch.outer(v1, v2)
# print(result) 
# # tensor([[ 4.,  5.,  6.],
# #         [ 8., 10., 12.],
# #         [12., 15., 18.]])


# -- concatenation --
# t1 = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[13,14, 15], [16,17,18]]])
# t2 = torch.tensor([[[7,8,9],[10,11,12]], [[19,20,21],[22,23,24]]])
# result=torch.cat((t1,t2),dim=0)
# print(result)
# result2 = torch.cat((t1,t2),dim=1)
# print(result2)
# result2 = torch.cat((t1,t2),dim=-1)
# print(result2)

# -- unsqueeze --
# t1 = torch.Tensor([1,2,3])
# print(t1.shape) # torch.Size([3])
# t2 = t1.unsqueeze(0)
# print(t2) # tensor([[1., 2., 3.]])
# print(t2.shape) # torch.Size([1, 3])