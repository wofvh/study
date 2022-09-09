from numpy import dtype
import torch
print(torch.__version__)
# 1.12.1

x = torch.empty(4,2)
print(x)
# tensor([[0., 0.],
#         [0., 0.],
#         [0., 0.],
#         [0., 0.]])

x = torch.rand(4,2)
print(x)
# tensor([[9.3239e-01, 7.0426e-01],
#         [5.1210e-01, 2.5439e-04],
#         [9.6842e-01, 7.3715e-01],
#         [8.2300e-02, 1.0708e-01]])

x = torch.zeros(4,2,dtype=torch.long)
print(x)
# tensor([[0, 0],
#         [0, 0],
#         [0, 0],
#         [0, 0]])

x= x.new_ones(2,4,dtype=torch.double)
print(x)
# tensor([[1., 1., 1., 1.],
#         [1., 1., 1., 1.]], dtype=torch.float64)

x = torch.randn_like(x,dtype=torch.float)
print(x)
# tensor([[-0.1394, -0.0447, -1.1807, -1.1572],
#         [-0.3656, -0.6502,  0.6889,  0.9801]])

print(x.size())
# torch.Size([2, 4])

ft = torch.FloatTensor([1,2,3])
print(ft)
print(ft.dtype)
# tensor([1., 2., 3.])
# torch.float32

print(ft.short())
print(ft.int())
print(ft.long())
# tensor([1, 2, 3], dtype=torch.int16)
# tensor([1, 2, 3], dtype=torch.int32)
# tensor([1, 2, 3])

it = torch.IntTensor([1,2,3])
print(it)
print(it.dtype)
# tensor([1, 2, 3], dtype=torch.int32)
# torch.int32

print(it.float())
print(it.double())
print(it.half())
# tensor([1., 2., 3.])
# tensor([1., 2., 3.], dtype=torch.float64)
# tensor([1., 2., 3.], dtype=torch.float16)

x= torch.randn(1)
print(x)
print(x.item())
print(x.dtype)
# tensor([0.8612])
# 0.8612411022186279
# torch.float32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
y= torch.ones_like(x,device=device)
print(y)
x= x.to(device)
print(x)
z = x +y 
print(z)
print(z.to('cpu',torch.double))

# cuda
# tensor([1.], device='cuda:0')
# tensor([-1.6460], device='cuda:0')
# tensor([-0.6460], device='cuda:0')
# tensor([-0.6460], dtype=torch.float64)

