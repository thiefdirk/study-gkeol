from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_diabetes, load_boston
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
dataset = load_boston()
x = dataset.data
y = dataset['target']

x = torch.FloatTensor(x)
y = torch.FloatTensor(y).unsqueeze(1)




from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,
                                                    train_size=0.8,shuffle=True,
                                                    random_state=66)


from sklearn.preprocessing import StandardScaler, MinMaxScaler
# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
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
#     nn.Linear(13,100),
#     nn.ReLU(),
#     nn.Linear(100,200),
#     nn.ReLU(),
#     nn.Linear(200,150),
#     nn.ReLU(),
#     nn.Linear(150,50),
#     nn.ReLU(),
#     nn.Linear(50,1)).to(DEVICE)

class BostonModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(13,100)
        self.linear2 = nn.Linear(100,200)
        self.linear3 = nn.Linear(200,150)
        self.linear4 = nn.Linear(150,50)
        self.linear5 = nn.Linear(50,1)
    def forward(self,x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = self.linear5(x)
        return x
    
model = BostonModel().to(DEVICE)

#3. 컴파일,훈련

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

def train(model,criterion,optimizer,train_loader):
    model.train()
    loss = 0
    for batch_idx,(data,target) in enumerate(train_loader):
        data = data.to(DEVICE)
        target = target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output,target)
        loss.backward()
        optimizer.step()
        loss += loss.item()
    return loss/len(train_loader)

def r2_score(y_test,y_pred):
    u = ((y_test-y_pred)**2).sum()
    v = ((y_test-y_test.mean())**2).sum()
    return 1-u/v

epochs = 1000
for epoch in range(1, epochs+1):
    loss = train(model,criterion,optimizer,train_loader)
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Loss: {:.6f}'.format(
            epoch, epochs, loss
        ))
        
#4. 평가,예측

def evaluate(model,criterion,test_loader):
    model.eval()
    loss = 0
    with torch.no_grad():
        for data,target in test_loader:
            data = data.to(DEVICE)
            target = target.to(DEVICE)
            output = model(data)
            loss += criterion(output,target).item()
    return loss/len(test_loader)


loss = evaluate(model,criterion,test_loader)

print('Loss: {:.6f}'.format(loss))
print('r2_score: {:.4f}%'.format(r2_score(y_test,model(x_test))))


# Loss: 15.357096
# r2_score: 0.8163%