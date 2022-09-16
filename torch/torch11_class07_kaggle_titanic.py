from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder
import time
from sklearn.datasets import load_breast_cancer, load_digits
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

###########################폴더 생성시 현재 파일명으로 자동생성###########################################
import inspect, os
a = inspect.getfile(inspect.currentframe()) #현재 파일이 위치한 경로 + 현재 파일 명
print(a)
print(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))) #현재 파일이 위치한 경로
print(a.split("\\")[-1]) #현재 파일 명
current_name = a.split("\\")[-1]
##########################밑에 filepath경로에 추가로  + current_name + '/' 삽입해야 돌아감#######################


#1. 데이터
path = './_data/kaggle_titanic/'
train_set = pd.read_csv(path + 'train.csv', # + 명령어는 문자를 앞문자와 더해줌
                        index_col=0)
# print(train_set)
# print(train_set.shape) # (891, 11)
# print(train_set.describe())
# print(train_set.columns)

test_set = pd.read_csv(path + 'test.csv', # 예측에서 쓸거임                
                       index_col=0)
# print(test_set)
# print(test_set.shape) # (418, 10)
# print(test_set.describe())

print(train_set.Pclass.value_counts())

Pclass1 = train_set["Survived"][train_set["Pclass"] == 1].value_counts(normalize = True)[1]*100
Pclass2 = train_set["Survived"][train_set["Pclass"] == 2].value_counts(normalize = True)[1]*100
Pclass3 = train_set["Survived"][train_set["Pclass"] == 3].value_counts(normalize = True)[1]*100
print(f"Percentage of Pclass 1 who survived: {Pclass1}")
print(f"Percentage of Pclass 2 who survived: {Pclass2}")
print(f"Percentage of Pclass 3 who survived: {Pclass3}")

female = train_set["Survived"][train_set["Sex"] == 'female'].value_counts(normalize = True)[1]*100
male = train_set["Survived"][train_set["Sex"] == 'male'].value_counts(normalize = True)[1]*100
print(f"Percentage of females who survived: {female}")
print(f"Percentage of males who survived: {male}")



#### 결측치 처리 1. 제거 ####

train_set = train_set.fillna({"Embarked": "C"})
train_set.Age = train_set.Age.fillna(value=train_set.Age.mean())

train_set = train_set.drop(['Name'], axis = 1)
test_set = test_set.drop(['Name'], axis = 1)

train_set = train_set.drop(['Ticket'], axis = 1)
test_set = test_set.drop(['Ticket'], axis = 1)

train_set = train_set.drop(['Cabin'], axis = 1)
test_set = test_set.drop(['Cabin'], axis = 1)

train_set = pd.get_dummies(train_set,drop_first=True)
test_set = pd.get_dummies(test_set,drop_first=True)

test_set.Age = test_set.Age.fillna(value=test_set.Age.mean())
test_set.Fare = test_set.Fare.fillna(value=test_set.Fare.mode())

print(train_set, test_set, train_set.shape, test_set.shape)

############################


x = train_set.drop(['Survived'], axis=1)
print(x)
print(x.columns)
print(x.shape) # (891, 8)

y = train_set['Survived']
print(y)
print(y.shape) # (891,)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=66
                                                    )

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_set = scaler.transform(test_set)
y_train = y_train.values # values : numpy로 변환
y_test = y_test.values

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)
test_set = torch.FloatTensor(test_set).to(DEVICE)


#2. 모델

# model = nn.Sequential(
#     nn.Linear(8,64),
#     nn.ReLU(),
#     nn.Linear(64,64),
#     nn.ReLU(),
#     nn.Linear(64,64),
#     nn.ReLU(),
#     nn.Linear(64,32),
#     nn.ReLU(),
#     nn.Linear(32,1)).to(DEVICE)

class TitanicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(8,64)
        self.linear2 = nn.Linear(64,64)
        self.linear3 = nn.Linear(64,64)
        self.linear4 = nn.Linear(64,32)
        self.linear5 = nn.Linear(32,1)

    def forward(self,x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = torch.sigmoid(self.linear5(x))
        return x
    
model = TitanicModel().to(DEVICE)

#3. 컴파일,훈련

criterion = nn.BCEWithLogitsLoss() # BCELoss : 이진분류, BCEWithLogitsLoss : 이진분류 + Sigmoid
optimizer = optim.Adam(model.parameters(),lr=0.0004)

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
