import torch

cls = torch.Tensor([[1, 2, 3]])
xywh = torch.Tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
conf = torch.Tensor([0.1, 0.2, 0.3])

line = (cls, xywh, conf) # tuple
print(line)

print(cls.shape)
cls = cls.view(-1)
print(cls.shape)

print(xywh.shape)
xywh = torch.tensor(xywh.shape)[[1, 0, 1, 0]] # [[1, 0, 1, 0]] is a trick to swap the shape
print(xywh.shape)
# tensor