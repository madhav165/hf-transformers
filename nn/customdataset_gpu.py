import os
import pandas as pd
from torch.utils.data import Dataset
import torch
import logging

class CustomDataset(Dataset):
    def __init__(self, flist, device, transform=None, target_transform=None):
        self.flist = flist
        self.df = pd.DataFrame()
        self.device = device
        for fl in self.flist:
            logging.info(f'Reading {fl}.')
            df_fl = pd.read_csv(fl)
            self.df = pd.concat([self.df, df_fl], axis=0)

        # self.df = self.df.iloc[:1000]
        
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data_detail = str(self.df.iloc[idx]['SIV_LIT_DSC_TE'])
        data_general = str(self.df.iloc[idx]['GOODSDESCRPTN'])
        try:
            digit_0 = torch.zeros(10, dtype=torch.float, device=self.device).scatter_(0, torch.tensor(int(str(self.df.iloc[idx]['TARIFFNUMBER'])[0]), device=self.device), value=1)
        except:
            digit_0 = torch.zeros(10, dtype=torch.float, device=self.device).scatter_(0, torch.tensor(0, device=self.device), value=1)
        digit_0 = digit_0[None,:]
        try:
            digit_1 = torch.zeros(10, dtype=torch.float, device=self.device).scatter_(0, torch.tensor(int(str(self.df.iloc[idx]['TARIFFNUMBER'])[1]), device=self.device), value=1)
        except:
            digit_1 = torch.zeros(10, dtype=torch.float, device=self.device).scatter_(0, torch.tensor(0, device=self.device), value=1)
        digit_1 = digit_1[None,:]
        try:
            digit_2 = torch.zeros(10, dtype=torch.float, device=self.device).scatter_(0, torch.tensor(int(str(self.df.iloc[idx]['TARIFFNUMBER'])[2]), device=self.device), value=1)
        except:
            digit_2 = torch.zeros(10, dtype=torch.float, device=self.device).scatter_(0, torch.tensor(0, device=self.device), value=1)
        digit_2 = digit_2[None,:]
        try:
            digit_3 = torch.zeros(10, dtype=torch.float, device=self.device).scatter_(0, torch.tensor(int(str(self.df.iloc[idx]['TARIFFNUMBER'])[3]), device=self.device), value=1)
        except:
            digit_3 = torch.zeros(10, dtype=torch.float, device=self.device).scatter_(0, torch.tensor(0, device=self.device), value=1)
        digit_3 = digit_3[None,:]
        try:
            digit_4 = torch.zeros(10, dtype=torch.float, device=self.device).scatter_(0, torch.tensor(int(str(self.df.iloc[idx]['TARIFFNUMBER'])[4]), device=self.device), value=1)
        except:
            digit_4 = torch.zeros(10, dtype=torch.float, device=self.device).scatter_(0, torch.tensor(0, device=self.device), value=1)
        digit_4 = digit_4[None,:]
        try:
            digit_5 = torch.zeros(10, dtype=torch.float, device=self.device).scatter_(0, torch.tensor(int(str(self.df.iloc[idx]['TARIFFNUMBER'])[5]), device=self.device), value=1)
        except:
            digit_5 = torch.zeros(10, dtype=torch.float, device=self.device).scatter_(0, torch.tensor(0, device=self.device), value=1)
        digit_5 = digit_5[None,:]
        try:
            digit_6 = torch.zeros(10, dtype=torch.float, device=self.device).scatter_(0, torch.tensor(int(str(self.df.iloc[idx]['TARIFFNUMBER'])[6]), device=self.device), value=1)
        except:
            digit_6 = torch.zeros(10, dtype=torch.float, device=self.device).scatter_(0, torch.tensor(0, device=self.device), value=1)
        digit_6 = digit_6[None,:]
        try:
            digit_7 = torch.zeros(10, dtype=torch.float, device=self.device).scatter_(0, torch.tensor(int(str(self.df.iloc[idx]['TARIFFNUMBER'])[7]), device=self.device), value=1)
        except:
            digit_7 = torch.zeros(10, dtype=torch.float, device=self.device).scatter_(0, torch.tensor(0, device=self.device), value=1)
        digit_7 = digit_7[None,:]
        try:
            digit_8 = torch.zeros(10, dtype=torch.float, device=self.device).scatter_(0, torch.tensor(int(str(self.df.iloc[idx]['TARIFFNUMBER'])[8]), device=self.device), value=1)
        except:
            digit_8 = torch.zeros(10, dtype=torch.float, device=self.device).scatter_(0, torch.tensor(0, device=self.device), value=1)
        digit_8 = digit_8[None,:]
        try:
            digit_9 = torch.zeros(10, dtype=torch.float, device=self.device).scatter_(0, torch.tensor(int(str(self.df.iloc[idx]['TARIFFNUMBER'])[9]), device=self.device), value=1)
        except:
            digit_9 = torch.zeros(10, dtype=torch.float, device=self.device).scatter_(0, torch.tensor(0, device=self.device), value=1)
        digit_9 = digit_9[None,:]
        try:
            label_0 = torch.tensor([int(str(self.df.iloc[idx]['PRMRYTARNR'])[0])], device=self.device)
        except:
            label_0 = torch.tensor([0], device=self.device)
        try:
            label_1 = torch.tensor([int(str(self.df.iloc[idx]['PRMRYTARNR'])[1])], device=self.device)
        except:
            label_1 = torch.tensor([0], device=self.device)
        try:
            label_2 = torch.tensor([int(str(self.df.iloc[idx]['PRMRYTARNR'])[2])], device=self.device)
        except:
            label_2 = torch.tensor([0], device=self.device)
        try:
            label_3 = torch.tensor([int(str(self.df.iloc[idx]['PRMRYTARNR'])[3])], device=self.device)
        except:
            label_3 = torch.tensor([0], device=self.device)
        try:
            label_4 = torch.tensor([int(str(self.df.iloc[idx]['PRMRYTARNR'])[4])], device=self.device)
        except:
            label_4 = torch.tensor([0], device=self.device)
        try:
            label_5 = torch.tensor([int(str(self.df.iloc[idx]['PRMRYTARNR'])[5])], device=self.device)
        except:
            label_5 = torch.tensor([0], device=self.device)
        try:
            label_6 = torch.tensor([int(str(self.df.iloc[idx]['PRMRYTARNR'])[6])], device=self.device)
        except:
            label_6 = torch.tensor([0], device=self.device)
        try:
            label_7 = torch.tensor([int(str(self.df.iloc[idx]['PRMRYTARNR'])[7])], device=self.device)
        except:
            label_7 = torch.tensor([0], device=self.device)
        try:
            label_8 = torch.tensor([int(str(self.df.iloc[idx]['PRMRYTARNR'])[8])], device=self.device)
        except:
            label_8 = torch.tensor([0], device=self.device)
        try:
            label_9 = torch.tensor([int(str(self.df.iloc[idx]['PRMRYTARNR'])[9])], device=self.device)
        except:
            label_9 = torch.tensor([0], device=self.device)
        if self.transform:
            data = torch.cat([self.transform(data_general), self.transform(data_detail), digit_0, digit_1, digit_2, digit_3, digit_4, digit_5, digit_6, digit_7, digit_8, digit_9], dim=1)
        if self.target_transform:
            label_0 = self.target_transform(label_0)
            label_1 = self.target_transform(label_1)
            label_2 = self.target_transform(label_2)
            label_3 = self.target_transform(label_3)
            label_4 = self.target_transform(label_4)
            label_5 = self.target_transform(label_5)
            label_6 = self.target_transform(label_6)
            label_7 = self.target_transform(label_7)
            label_8 = self.target_transform(label_8)
            label_9 = self.target_transform(label_9)
        return data, label_0, label_1, label_2, label_3, label_4, label_5, label_6, label_7, label_8, label_9