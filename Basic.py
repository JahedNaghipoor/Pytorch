import torch

# Tensor initialization
device = 'cuda' if torch.cuda.is_available() else 'cpu'

my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32,
                         device=device, requires_grad=True)
print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.shape)
print(my_tensor.requires_grad)

# un-initialize data
x1 = torch.empty(size=(3, 3))
x2 = torch.zeros(size=(3, 3))
x3 = torch.rand(size=(3, 3))
x4 = torch.ones(size=(3, 3))
x5 = torch.eye(3, 3)
x6 = torch.arange(start=0, end=5, step=1)
x7 = torch.linspace(start=0.1, end=1, steps=10)
x8 = torch.empty(size=(1, 5)).normal_(mean=0, std=1)
x9 = torch.empty(size=(1, 5)).uniform_(0, 1)
x10 = torch.diag(torch.ones(3))
# print(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10)

tensor = torch.arange(4)
print(tensor.short())  # int16
print(tensor.long())  # int64
print(tensor.half())  # float16
print(tensor.float())  # float32
print(tensor.double())  # float64

# array to tensor and vice-versa
import numpy as np

np_array = np.array((5, 5))
tensor = torch.from_numpy(np_array)
print(tensor)
np_array_back = tensor.numpy()
print(np_array_back)

# torch math and operations

x = torch.tensor([-1, 2, 3])
y = torch.tensor([9, 8, 7])

# addition and subtraction
z1 = torch.empty(3)
torch.add(x, y, out=z1)
z2 = torch.add(x, y)
z3 = x + y
z4 = x - y
print(z1, z2, z3, z4)

# division
z = torch.true_divide(x, y)

# inplace operations
t = torch.zeros(3)
t.add_(x)  # or t += x

# exponentiation
z = x.pow(2)  # or x**2

# matrix multiplication

x1 = torch.rand((2, 5))
x2 = torch.rand((5, 4))
x3 = torch.mm(x1, x2)
x3 = x1.mm(x2)

# matrix exponentiation
matrix_exp = torch.rand((5, 5))
matrix_exp.matrix_power(3)
print(matrix_exp)

# element-wise multiplication
z = x * y
# dot product
z = torch.dot(x, y)

# batch Matrix Multiplication
batch = 32
n = 10
m = 20
p = 30
tensor1 = torch.rand(batch, n, m)
tensor2 = torch.rand(batch, m, p)
out_bmm = torch.bmm(tensor1, tensor2)  # (batch, n, p)
# print(out_bmm)

# Example of broadcasting
x1 = torch.rand((5, 5))
x2 = torch.rand((1, 5))

z = x1 - x2  # it works in pytorch and numpy
sum_x = torch.sum(x, dim=0)
values, indices = torch.max(x, dim=0) # or x.max(dim=0)
abs_x = torch.abs(x)
z = torch.argmax(x, dim=0)  # and argmin
mean_x = torch.mean(x.float(), dim=0)

z = torch.eq(x, y)

sorted_y, indices = torch.sort(y, dim=0, descending=False)

z = torch.clamp(x, min=0)
print(z)

x = torch.tensor([1, 0, 1, 1], dtype=torch.bool)
z = torch.all(x) # torch.any()
print(z)

# indexing in tensor

batch_size = 10
features = 25

x = torch.rand(batch_size, features)
print(x[0].shape) # or x[0,:].shape
print(x[:, 0].shape)

print(x[2, 0:10])

# Fancy indexing
x = torch.arange(10)
indices = [2, 5, 8]
print(x[indices])

x = torch.rand((3, 5))
rows = torch.tensor([1, 0])
cols = torch.tensor([4, 0])
print(x[rows, cols])

# advanced indexing
x = torch.arange(10)
print(x[(x < 2) | (x > 8)])
print(x[x.remainder(2) == 0])

print(torch.where(x > 5, 10-x, x))
print(torch.tensor([0, 0, 1, 3, 5, 6]).unique())
print(x.ndimension())
print(x.numel()) # dimension of tensor


# reshape a tensor
x = torch.arange(9)
x_3x3 = x.view(3, 3)
x_3x3 = x.reshape(3, 3)
print(x_3x3)

y = x_3x3.t() #transpose
print(y)

x1 = torch.rand((2, 5))
x2 = torch.rand((2, 5))

print(torch.cat((x1, x2), dim=0).shape)
print(torch.cat((x1, x2), dim=1).shape)

z = x1.view(-1) # flatten

x1 = torch.rand((batch, 2, 5))
z = x1.view(batch, -1)
print(z.shape)

# add/remove dimension
x = torch.arange(10)
print(x.unsqueeze(0).shape)
print(x.unsqueeze(1).shape)
print(x.squeeze(0).shape)




