import torch
import torch.nn as nn

# # -- dropout --
# dropout_layer = nn.Dropout(p=0.5)

# t1 = torch.Tensor([1,2,3])
# t2 = dropout_layer(t1)

# print(t2) # to keep the same expectation the remaining elements should be scaled by 1/(1-p) = 1/(1-0.5) = 2
# # tensor([2., 0., 6.]) # the non-dropped elements are scaled by 2, and the dropped element is set to 0.


# # -- linear layer --
# layer = nn.Linear(in_features=3, out_features=5, bias=True)
# t1 = torch.Tensor([1,2,3]) # input tensor of shape (3,)

# t2 = torch.Tensor([[1,2,3]]) # input tensor of shape (1, 3) - batch size of 1

# output = layer(t1) # output tensor of shape (1, 5)
# print(output)

# # -- view and reshape --
# t = torch.Tensor([[1,2,3,4,5,6], [7,8,9,10,11,12]])
# print(t.shape) # torch.Size([2, 6])
# t_view1 = t.view(3, 4) # 3 rows, 4 columns
# print(t_view1)
# t_view2 = t.view(4, 3) # 4 rows, 3 columns
# print(t_view2)

# # -- transpose --
# t1 = torch.Tensor([[1, 2, 3], [4, 5, 6]])
# t1 = t1.transpose(0, 1) # swap the dimensions 0 and 1
# print(t1) # tensor([[1., 4.], [2., 5.], [3., 6.]])

# # -- upper and lower triangular --
# x = torch.tensor([[1,2,3], [4,5,6], [7,8,9]]) # a 3x3 matrix
# print(torch.triu(x)) # tensor([[1, 2, 3], [0, 5, 6], [0, 0, 9]]), diagpnal is set to 1 dy default
# # triu stands for "upper triangular".

# print(torch.tril(x, diagonal=1)) # tensor([[1, 2, 0], [4, 5, 6], [7, 8, 9]]),
# # diagonal=1 means include the first diagonal above the main diagonal.

# # -- reshape --
# x = torch.arange(1,7)
# y =torch.reshape(x,(2,3))
# print(y) # tensor([[1, 2, 3], [4, 5, 6]])

# # use -1 to automatically infer the size of the dimension
# z = torch.reshape(x, (3, -1))
# print(z) # tensor([[1, 2], [3, 4], [5, 6]])

# # .view() → fast, strict, no copy, requires contiguous memory
# # reshape() → flexible, may copy, safer in real-world code

#
