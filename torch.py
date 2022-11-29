import torch

cls = torch.Tensor([1, 2, 3])
xywh = torch.Tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
conf = torch.Tensor([0.1, 0.2, 0.3])

line = (cls, xywh, conf)
print(line)

# tensor