from sklearn.datasets import load_breast_cancer
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

#1. 데이터
dataset = load_breast_cancer()
x = dataset.data
y = dataset['target']

x = torch.FloatTensor(x)
y = torch.FloatTensor(y).unsqueeze(1)


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,
                                                    train_size=0.8,shuffle=True,
                                                    random_state=66, stratify=y)


from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.FloatTensor(y_train).to(DEVICE)
y_test = torch.FloatTensor(y_test).to(DEVICE)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

#2. 모델

model = nn.Sequential(
    nn.Linear(30,64),
    nn.ReLU(),
    nn.Linear(64,64),
    nn.ReLU(),
    nn.Linear(64,64),
    nn.ReLU(),
    nn.Linear(64,32),
    nn.ReLU(),
    nn.Linear(32,1)).to(DEVICE)

#3. 컴파일,훈련

criterion = nn.BCEWithLogitsLoss() # BCELoss : 이진분류, BCEWithLogitsLoss : 이진분류 + Sigmoid
optimizer = optim.Adam(model.parameters(),lr=0.001)

def train(model,criterion,optimizer,x_train,y_train):
    model.train()
    optimizer.zero_grad()
    prediction = model(x_train)
    loss = criterion(prediction,y_train)
    loss.backward()
    optimizer.step()
    return loss.item()

def binary_acc(prediction,y_test):
    prediction = torch.round(torch.sigmoid(prediction))
    correct_result_sum = (prediction == y_test).sum().float() # 예측값과 실제값이 같은 것들의 합, .sum() : 합계, .float() : 실수형으로 변환
    acc = correct_result_sum/prediction.size(0) # prediction.size(0) : 행의 개수
    acc = (acc*100)
    return acc

epochs = 1000
for epoch in range(1, epochs+1):
    loss = train(model,criterion,optimizer,x_train,y_train)
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Loss: {:.6f}/ Accuracy: {:.4f}%'.format(
            epoch,epochs,loss,binary_acc(model(x_test),y_test)))
        
#4. 평가,예측

def evaluate(model,criterion,x_test,y_test):
    model.eval()
    with torch.no_grad():
        prediction = model(x_test)
        loss = criterion(prediction,y_test)
    return loss.item()


loss = evaluate(model,criterion,x_test,y_test)

print('Loss: {:.6f}'.format(loss))
print('Accuracy: {:.4f}%'.format(binary_acc(model(x_test),y_test)))

from sklearn.metrics import accuracy_score

y_pred = torch.round(torch.sigmoid(model(x_test)))
y_pred = y_pred.cpu().detach().numpy()
y_test = y_test.cpu().detach().numpy()
print('Accuracy: {:.4f}%'.format(accuracy_score(y_test,y_pred)*100))
