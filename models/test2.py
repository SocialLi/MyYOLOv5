import torch

# nx, ny = 4, 3
# y, x = torch.arange(ny), torch.arange(nx)
# yv, xv = torch.meshgrid(y, x, indexing='ij')
#
# shape = 1, 3, ny, nx, 2
#
# res = torch.stack((xv, yv), 2)
# print(res)
# print(res.shape)
#
# res2 = res.expand(shape) - 0.5
#
# print(res2)
# print(res2.shape)

x = torch.arange(24).reshape(2, 3, 2, 2)
y = torch.arange(36).reshape(3, 3, 2, 2)
z = torch.cat((x, y), 1)
print(z.shape)
