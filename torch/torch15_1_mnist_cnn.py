from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision.transforms as tr

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')


transf = tr.Compose([tr.Resize((150)), tr.ToTensor()]) # Compose : 여러개의 transform을 하나로 묶어준다.

path = './_data/torch_data/'
# train_dataset = MNIST(path, train=True, download=True, transform=transf)
# test_dataset = MNIST(path, train=False, download=True, transform=transf)
train_dataset = MNIST(path, train=True, download=False)
test_dataset = MNIST(path, train=False, download=False)


x_train, y_train = train_dataset.data/255., train_dataset.targets
x_test, y_test = test_dataset.data/255., test_dataset.targets

print(x_train.shape, y_train.shape)
print(x_test.size(), y_test.size())

print(np.min(x_train.numpy()), np.max(x_train.numpy()))

# x_train, x_test = x_train.view(-1, 28*28), x_test.reshape(-1, 28*28) # torch, tensor 차이점 : 60000, 28,28,1 -> 60000, 1, 28, 28
x_train, x_test = x_train.unsqueeze(1), x_test.unsqueeze(1) 
print(x_train.shape, x_test.shape) # torch.Size([60000, 1, 28, 28]) torch.Size([10000, 1, 28, 28])

train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

class CNN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        
        self.hidden_layer1 = nn.Sequential(
        nn.Conv2d(num_features, 64, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Dropout2d(0.5))
        
        self.hidden_layer2 = nn.Sequential(
        nn.Conv2d(64, 32, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Dropout2d(0.5))

        self.hidden_layer3 = nn.Flatten()
        
        self.hidden_layer4 = nn.Sequential(
            nn.Linear(32*5*5, 100),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        self.output_layer = nn.Sequential(
            nn.Linear(100, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.hidden_layer3(x)
        x = self.hidden_layer4(x)
        x = self.output_layer(x)
        return x
    
model = CNN(1).to(DEVICE)

criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, train_loader, optimizer, criterion):
    model.train()
    loss_list = 0
    correct_list = 0
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        prediction = torch.argmax(output, dim=1)
        acc = (prediction == y).float().mean()
        loss_list += loss.item()
        correct_list += acc.item()
    return loss_list/len(train_loader), correct_list/len(train_loader)

def evaluate(model, test_loader, criterion):
    model.eval() # dropout, batch normalization을 평가모드로 바꿔준다.
    loss_list = 0
    correct_list = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            output = model(x)
            loss = criterion(output, y)
            prediction = torch.argmax(output, dim=1)
            acc = (prediction == y).float().mean()
            loss_list += loss.item()
            correct_list += acc.item()
        return loss_list/len(test_loader), correct_list/len(test_loader)
    
epochs = 20
for epoch in range(epochs):
    loss, acc = train(model, train_loader, optimizer, criterion)
    val_loss, val_acc = evaluate(model, test_loader, criterion)
    print(f'epoch : {epoch+1}, loss : {loss:.3f}, accuracy : {acc:.3f}, val_loss : {val_loss:.3f}, val_accuracy : {val_acc:.3f}')



loss, acc = evaluate(model, test_loader, criterion)

print(f'loss : {loss:.3f}, accuracy : {acc:.3f}')