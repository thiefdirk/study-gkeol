import random
import pandas as pd
import numpy as np
import os
import glob

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from multiprocessing import freeze_support
from statsmodels.tsa.arima.model import ARIMA # ARIMA 모델
from xgboost import XGBRegressor , XGBRFRegressor # XGBRFRegressor : 랜덤포레스트와 유사
#import convlstm

from tqdm.auto import tqdm

import warnings
warnings.filterwarnings(action='ignore') 

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

CFG = {
    'EPOCHS':200,
    'LEARNING_RATE':1e-3,
    'BATCH_SIZE':300,
    'SEED':123
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED']) # Seed 고정

paths = 'C:\study\_data\dacon_chung/'

all_input_list = sorted(glob.glob(paths +'train_input/*.csv'))
all_target_list = sorted(glob.glob(paths +'train_target/*.csv'))


train_input_list = all_input_list[:50]
train_target_list = all_target_list[:50]

val_input_list = all_input_list[50:]
val_target_list = all_target_list[50:]

class CustomDataset(Dataset):
    def __init__(self, input_paths, target_paths, infer_mode):
        self.input_paths = input_paths
        self.target_paths = target_paths
        self.infer_mode = infer_mode
        
        self.data_list = []
        self.label_list = []
        print('Data Pre-processing..')
        for input_path, target_path in tqdm(zip(self.input_paths, self.target_paths)):
            input_df = pd.read_csv(input_path)
            target_df = pd.read_csv(target_path)
            # 2021-02-17 00:00:00 -> 00:00:00
            input_df['시간'] = input_df['시간'].apply(lambda x : x.split(' ')[1]) 
            input_df['시간'] = input_df['시간'].apply(lambda x : x.split(':')[1])
            input_df['시간'] = input_df['시간'].astype(int)
            # minmax
            input_df = (input_df - input_df.min()) / (input_df.max() - input_df.min())
            input_df = input_df.fillna(0)
            
            input_length = int(len(input_df)/1440)
            target_length = int(len(target_df))
            
            for idx in range(target_length):
                time_series = input_df[1440*idx:1440*(idx+1)].values
                self.data_list.append(torch.Tensor(time_series))
            for label in target_df["rate"]:
                self.label_list.append(label)
        print('Done.')
        print()
              
    def __getitem__(self, index):
        data = self.data_list[index]
        label = self.label_list[index]
        if self.infer_mode == False:
            return data, label
        else:
            return data
            
    def __len__(self):
        return len(self.data_list)
    
train_dataset = CustomDataset(train_input_list, train_target_list, False)
train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

val_dataset = CustomDataset(val_input_list, val_target_list, False)
val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

all_dataset = CustomDataset(all_input_list, all_target_list, False)

class ConvLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, num_layers):
        super(ConvLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        
        self.conv = nn.Conv2d(self.input_size + self.hidden_size, 4*self.hidden_size, self.kernel_size, 1, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x, hidden_state): # hidden_state = (h, c), x = (batch, channel, height, width)
        h_cur, c_cur = hidden_state
        combined = torch.cat((x, h_cur), dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_size, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f*c_cur + i*g
        h_next = o*torch.tanh(c_next)
        return h_next, c_next
    
    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_size, height, width).to(device),
               torch.zeros(batch_size, self.hidden_size, height, width).to(device))

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=38, hidden_size=256, batch_first=True, bidirectional=False)
        self.lstm2 = nn.LSTM(input_size=256, hidden_size=128, batch_first=True, bidirectional=False)
        self.lstm3 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True, bidirectional=False)
        self.gru1 = nn.GRU(input_size=37, hidden_size=40, batch_first=True, bidirectional=False)
        self.gru2 = nn.GRU(input_size=40, hidden_size=60, batch_first=True, bidirectional=False)
        self.gru3 = nn.GRU(input_size=60, hidden_size=90, batch_first=True, bidirectional=False) # batch_first=True : (batch, seq, feature)
        self.convlstm = ConvLSTM(input_size=38, hidden_size=256, kernel_size=(3,3), num_layers=3)
        self.Flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(256, 1)
        )
        self.dropout = nn.Dropout(0.2)
        self.batchnorm = nn.BatchNorm1d(1440) # 1차원 데이터에 대한 BatchNorm
        self.dense1 = nn.Linear(64, 1)
        self.dense2 = nn.Linear(128, 1)
    def forward(self, x): # forward : 순전파
        # hidden, _ = self.gru1(x)
        # hidden = self.dropout(hidden)
        # hidden, _= self.gru2(hidden)
        # hidden = self.dropout(hidden)
        # hidden, _ = self.gru3(hidden)
        
        hidden = self.convlstm(x)
        output = self.classifier(hidden[:,-1,:])
        return output
    
class RMSE(nn.Module):
    def __init__(self):
        super(RMSE, self).__init__()
    def forward(self, y_pred, y_true):
        return torch.sqrt(torch.mean((y_pred-y_true)**2))
    
def train(model, optimizer, train_loader, val_loader, scheduler, device):
    model.to(device)
    # criterion = nn.MSELoss().to(device)
    # criterion = nn.L1Loss().to(device) # L1Loss : MAE
    criterion = RMSE().to(device)
    
    
    best_loss = 9999
    best_model = None
    for epoch in range(1, CFG['EPOCHS']+1):
        model.train()
        train_loss = []
        for X, Y in tqdm(iter(train_loader)):
            X = X.to(device)
            Y = Y.to(device)
            
            optimizer.zero_grad()
            
            output = model(X)
            loss = criterion(output, Y)
            
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
                    
        val_loss = validation(model, val_loader, criterion, device)
        
        print(f'Train Loss : [{np.mean(train_loss):.4f}] | Val Loss : [{val_loss:.4f}] | Epoch : [{epoch}/{CFG["EPOCHS"]}]')
        
        if scheduler is not None:
            scheduler.step(metrics=val_loss)
            
        if best_loss > val_loss:
            best_loss = val_loss
            best_model = model
    return best_model

def validation(model, val_loader, criterion, device):
    model.eval()
    val_loss = []
    with torch.no_grad():
        for X, Y in tqdm(iter(val_loader)):
            X = X.float().to(device)
            Y = Y.float().to(device)
            
            model_pred = model(X)
            loss = criterion(model_pred, Y)
            
            val_loss.append(loss.item())
            
    return np.mean(val_loss)

model = BaseModel()
model.eval() # 모델을 평가 모드로 설정
optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG["LEARNING_RATE"]) # params : 최적화할 모델의 파라미터
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30, verbose=True)

best_model = train(model, optimizer, train_loader, val_loader, scheduler, device)

test_input_list = sorted(glob.glob(paths + 'test_input/*.csv'))
test_target_list = sorted(glob.glob(paths + 'test_target/*.csv'))

def inference_per_case(model, test_loader, test_path, device):
    model.to(device)
    model.eval()
    pred_list = []
    with torch.no_grad():
        for X in iter(test_loader):
            X = X.float().to(device)
            
            model_pred = model(X)
            
            model_pred = model_pred.cpu().numpy().reshape(-1).tolist()
            
            pred_list += model_pred
    
    submit_df = pd.read_csv(test_path)
    submit_df['rate'] = pred_list
    submit_df.to_csv(test_path, index=False)
    
for test_input_path, test_target_path in zip(test_input_list, test_target_list):
    test_dataset = CustomDataset([test_input_path], [test_target_path], True)
    test_loader = DataLoader(test_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)
    inference_per_case(best_model, test_loader, test_target_path, device)
    
import zipfile
filelist = ['TEST_01.csv','TEST_02.csv','TEST_03.csv','TEST_04.csv','TEST_05.csv', 'TEST_06.csv']
os.chdir(paths + "test_target")
with zipfile.ZipFile(paths + "submission.zip", 'w') as my_zip:
    for i in filelist:
        my_zip.write(i)
    my_zip.close()
    
print('end')