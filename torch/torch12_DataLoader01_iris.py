from sklearn.datasets import load_breast_cancer, load_digits, load_iris
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# import to_categorical
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset, DataLoader, TensorDataset


USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

#1. 데이터
dataset = load_iris()
x = dataset.data
y = dataset['target']
print(y)

x = torch.FloatTensor(x)
y = torch.FloatTensor(y).unsqueeze(1)

y = OneHotEncoder().fit_transform(y).toarray()




from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,
                                                    train_size=0.8,shuffle=True,
                                                    random_state=66, stratify=y)


from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.FloatTensor(y_train).to(DEVICE)
y_test = torch.FloatTensor(y_test).to(DEVICE)

train_dataset = TensorDataset(x_train,y_train)
test_dataset = TensorDataset(x_test,y_test)

train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=32,shuffle=True)

#2. 모델

# model = nn.Sequential(
#     nn.Linear(4,32),
#     nn.ReLU(),
#     nn.Linear(32,16),
#     nn.ReLU(),
#     nn.Linear(16,8),
#     nn.ReLU(),
#     nn.Linear(8,4),
#     nn.ReLU(),
#     nn.Linear(4,3),
#     nn.Softmax()).to(DEVICE)

class IrisModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4,32)
        self.linear2 = nn.Linear(32,16)
        self.linear3 = nn.Linear(16,8)
        self.linear4 = nn.Linear(8,4)
        self.linear5 = nn.Linear(4,3)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    def forward(self,x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.relu(x)
        x = self.linear5(x)
        x = self.softmax(x)
        return x
    
model = IrisModel().to(DEVICE)

#3. 컴파일,훈련

criterion = nn.CrossEntropyLoss() # 이진분류에서는 BCELoss
optimizer = optim.Adam(model.parameters(),lr=0.001)

def train(model,criterion,optimizer,train_loader):
    model.train()
    loss = 0
    for batch_idx,(x_train,y_train) in enumerate(train_loader):
        optimizer.zero_grad()
        prediction = model(x_train)
        loss = criterion(prediction,y_train)
        loss.backward()
        optimizer.step()
        loss += loss.item()
    return loss


epochs = 1000
for epoch in range(1, epochs+1):
    loss = train(model,criterion,optimizer,train_loader)
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Loss: {:.6f}'.format(epoch,epochs,loss))
        
#4. 평가,예측

def evaluate(model,criterion,test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for x_test,y_test in test_loader:
            prediction = model(x_test)
            test_loss += criterion(prediction,y_test)
    test_loss /= len(test_loader.dataset)
    return test_loss


loss = evaluate(model,criterion,test_loader)

print('Loss: {:.6f}'.format(loss))

from sklearn.metrics import accuracy_score

y_pred = model(x_test)
y_pred = torch.argmax(y_pred,1)
y_test = torch.argmax(y_test,1)
print('Accuracy: {:.4f}%'.format(accuracy_score(y_test.cpu(),y_pred.cpu())*100))
