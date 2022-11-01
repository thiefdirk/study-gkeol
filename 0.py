import torch


a = torch.IntTensor(800)

# print(a)

for index, l in enumerate(a):
    print(index, l)
    for i in range(l+1):
        print(i)
        exit()