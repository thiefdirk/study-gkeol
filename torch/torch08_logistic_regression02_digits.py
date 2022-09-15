from sklearn.datasets import load_breast_cancer, load_digits
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# import to_categorical
from sklearn.preprocessing import OneHotEncoder

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

#1. 데이터
dataset = load_digits()
x = dataset.data
y = dataset['target']

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

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

#2. 모델

model = nn.Sequential(
    nn.Linear(64,32),
    nn.ReLU(),
    nn.Linear(32,16),
    nn.ReLU(),
    nn.Linear(16,8),
    nn.ReLU(),
    nn.Linear(8,4),
    nn.ReLU(),
    nn.Linear(4,10),
    nn.Softmax()).to(DEVICE)

#3. 컴파일,훈련

criterion = nn.CrossEntropyLoss() # 이진분류에서는 BCELoss
optimizer = optim.Adam(model.parameters(),lr=0.001)

def train(model,criterion,optimizer,x_train,y_train):
    model.train()
    optimizer.zero_grad()
    prediction = model(x_train)
    loss = criterion(prediction,y_train)
    loss.backward()
    optimizer.step()
    return loss.item()

def compute_accuracy(model,x_test,y_test):
    model.eval()
    with torch.no_grad():
        prediction = model(x_test)
        correct_prediction = torch.argmax(prediction,1) == torch.argmax(y_test,1)
        accuracy = correct_prediction.float().mean()
    return accuracy.item()

epochs = 1000
for epoch in range(1, epochs+1):
    loss = train(model,criterion,optimizer,x_train,y_train)
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Loss: {:.6f}/ Accuracy: {:.4f}%'.format(
            epoch,epochs,loss,compute_accuracy(model,x_test,y_test)))
        
#4. 평가,예측

def evaluate(model,criterion,x_test,y_test):
    model.eval()
    with torch.no_grad():
        prediction = model(x_test)
        loss = criterion(prediction,y_test)
    return loss.item()


loss = evaluate(model,criterion,x_test,y_test)

print('Loss: {:.6f}'.format(loss))
print('Accuracy: {:.4f}%'.format(compute_accuracy(model,x_test,y_test)))

from sklearn.metrics import accuracy_score

y_pred = model(x_test)
y_pred = torch.argmax(y_pred,1)
y_test = torch.argmax(y_test,1)
print('Accuracy: {:.4f}%'.format(accuracy_score(y_test.cpu(),y_pred.cpu())))
