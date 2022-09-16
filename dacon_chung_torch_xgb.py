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
from xgboost import XGBRegressor

from tqdm.auto import tqdm

import warnings
warnings.filterwarnings(action='ignore') 

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

CFG = {
    'EPOCHS':100,
    'LEARNING_RATE':1e-3,
    'BATCH_SIZE':128,
    'SEED':123,
    'LR' : 0.01,
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
            
            input_df = input_df.drop(columns=['시간'])
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

class XgbModel(nn.Module):
    def __init__(self):
        super(XgbModel, self).__init__()
        self.model = XGBRegressor()
        self.parameters = self.model.get_params() # 모델의 파라미터를 가져옴
        
    def forward(self, x):
        x = x.view(x.size(0), -1) # view : 텐서의 크기를 재구성
        x = self.model(x)
        return x
    
class ArimaModel(nn.Module):
    def __init__(self):
        super(ArimaModel, self).__init__()
        self.model = ARIMA()
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.model(x)
        return x
    
def xgb_train(train_loader, model, optimizer, criterion):
    model.train()
    train_loss = 0
    for data, label in tqdm(train_loader):
        data = data.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    return train_loss

def xgb_valid(val_loader, model, criterion):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, label in tqdm(val_loader):
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            loss = criterion(output, label)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    return val_loss



test_input_list = sorted(glob.glob(paths + 'test_input/*.csv'))
test_target_list = sorted(glob.glob(paths + 'test_target/*.csv'))

def xgb_infer(test_input_list, test_target_list, model):
    model.eval()
    test_dataset = CustomDataset(test_input_list, test_target_list, True)
    test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)
    with torch.no_grad():
        for data in tqdm(test_loader):
            data = data.to(device)
            output = model(data)
            output = output.cpu().numpy()
            output = output.reshape(-1, 1)
            output = pd.DataFrame(output)
            output.to_csv(paths + 'submission.csv', mode='a', header=False, index=False)

model = XgbModel()
optimizer = optim.Adam(model.parameters(), lr=CFG['LR']) # model.parameters() : 모델의 매개변수를 최적화
criterion = nn.L1Loss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

for test_input_path, test_target_path in zip(test_input_list, test_target_list):
    for epoch in range(CFG['EPOCHS']):
        train_loss = xgb_train(train_loader, model, optimizer, criterion)
        val_loss = xgb_valid(val_loader, model, criterion)
        scheduler.step(val_loss)
        print(f'Epoch {epoch+1} | train_loss : {train_loss:.4f} | val_loss : {val_loss:.4f}')
    xgb_infer(test_input_path, test_target_path, model)
    
    
import zipfile
filelist = ['TEST_01.csv','TEST_02.csv','TEST_03.csv','TEST_04.csv','TEST_05.csv', 'TEST_06.csv']
os.chdir(paths + "test_target")
with zipfile.ZipFile(paths + "submission_.zip", 'w') as my_zip:
    for i in filelist:
        my_zip.write(i)
    my_zip.close()
    
print('end')