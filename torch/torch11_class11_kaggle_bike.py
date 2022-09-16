from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_diabetes, load_boston, fetch_california_housing
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# import to_categorical
from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

###########################폴더 생성시 현재 파일명으로 자동생성###########################################
import inspect, os
a = inspect.getfile(inspect.currentframe()) #현재 파일이 위치한 경로 + 현재 파일 명
print(a)
print(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))) #현재 파일이 위치한 경로
print(a.split("\\")[-1]) #현재 파일 명
current_name = a.split("\\")[-1]
##########################밑에 filepath경로에 추가로  + current_name + '/' 삽입해야 돌아감#######################


#1. 데이터
path = './_data/kaggle_bike/'
train_set = pd.read_csv(path + 'train.csv') # + 명령어는 문자를 앞문자와 더해줌  index_col=n n번째 컬럼을 인덱스로 인식
            
test_set = pd.read_csv(path + 'test.csv') # 예측에서 쓸거임        



######## 년, 월 ,일 ,시간 분리 ############

train_set["hour"] = [t.hour for t in pd.DatetimeIndex(train_set.datetime)]
train_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(train_set.datetime)]
train_set["month"] = [t.month for t in pd.DatetimeIndex(train_set.datetime)]
train_set['year'] = [t.year for t in pd.DatetimeIndex(train_set.datetime)]
train_set['year'] = train_set['year'].map({2011:0, 2012:1})

test_set["hour"] = [t.hour for t in pd.DatetimeIndex(test_set.datetime)]
test_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(test_set.datetime)]
test_set["month"] = [t.month for t in pd.DatetimeIndex(test_set.datetime)]
test_set['year'] = [t.year for t in pd.DatetimeIndex(test_set.datetime)]
test_set['year'] = test_set['year'].map({2011:0, 2012:1})

train_set.drop('datetime',axis=1,inplace=True) # 트레인 세트에서 데이트타임 드랍
train_set.drop('casual',axis=1,inplace=True) # 트레인 세트에서 캐주얼 레지스터드 드랍
train_set.drop('registered',axis=1,inplace=True)

test_set.drop('datetime',axis=1,inplace=True) # 트레인 세트에서 데이트타임 드랍

##########################################


x = train_set.drop(['count'], axis=1)

y = train_set['count'] 

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

x = torch.FloatTensor(x.values)
y = torch.FloatTensor(y.values).unsqueeze(1)




from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,
                                                    train_size=0.8,shuffle=True,
                                                    random_state=66)


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

# model = nn.Sequential(
#     nn.Linear(12,100),
#     nn.ReLU(),
#     nn.Linear(100,200),
#     nn.ReLU(),
#     nn.Linear(200,150),
#     nn.ReLU(),
#     nn.Linear(150,50),
#     nn.ReLU(),
#     nn.Linear(50,1)).to(DEVICE)


class BikeModel(nn.Module):
    def __init__(self):
        super(BikeModel,self).__init__()
        self.fc1 = nn.Linear(12,100)
        self.fc2 = nn.Linear(100,200)
        self.fc3 = nn.Linear(200,150)
        self.fc4 = nn.Linear(150,50)
        self.fc5 = nn.Linear(50,1)
        self.relu = nn.ReLU()
    def forward(self,x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    
model = BikeModel().to(DEVICE)

#3. 컴파일,훈련

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

def train(model,criterion,optimizer,x_train,y_train):
    model.train()
    optimizer.zero_grad()
    prediction = model(x_train)
    loss = criterion(prediction,y_train)
    loss.backward()
    optimizer.step()
    return loss.item()

def r2_score(y_test,y_pred):
    u = ((y_test-y_pred)**2).sum()
    v = ((y_test-y_test.mean())**2).sum()
    return 1-u/v

epochs = 1000
for epoch in range(1, epochs+1):
    loss = train(model,criterion,optimizer,x_train,y_train)
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Loss: {:.6f} r2_score: {:.4f}'.format(
            epoch,epochs,loss,r2_score(y_test,model(x_test))))
        
#4. 평가,예측

def evaluate(model,criterion,x_test,y_test):
    model.eval()
    with torch.no_grad():
        prediction = model(x_test)
        loss = criterion(prediction,y_test)
    return loss.item()


loss = evaluate(model,criterion,x_test,y_test)

print('Loss: {:.6f}'.format(loss))
print('r2_score: {:.4f}'.format(r2_score(y_test,model(x_test))))


# Loss: 5536.333008
# r2_score: 0.8248