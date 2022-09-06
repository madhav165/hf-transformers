import os
import pandas as pd
from torch.utils.data import Dataset
import torch

class CustomDataset_6Separate(Dataset):
    def __init__(self, file_dir, transform=None, target_transform=None):
        self.flist = []
        for (dirpath, dirnames, filenames) in os.walk('./inputvectors/nldata'):
            self.flist.extend(filenames)
            break
        self.df = pd.DataFrame()
        for fl in self.flist:
            df_fl = pd.read_csv(os.path.join('inputvectors', 'nldata', fl))
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
            digit_0 = torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(int(str(self.df.iloc[idx]['TARIFFNUMBER'])[0])), value=1)
        except:
            digit_0 = torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(0), value=1)
        digit_0 = digit_0[None,:]
        try:
            digit_1 = torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(int(str(self.df.iloc[idx]['TARIFFNUMBER'])[1])), value=1)
        except:
            digit_1 = torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(0), value=1)
        digit_1 = digit_1[None,:]
        try:
            digit_2 = torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(int(str(self.df.iloc[idx]['TARIFFNUMBER'])[2])), value=1)
        except:
            digit_2 = torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(0), value=1)
        digit_2 = digit_2[None,:]
        try:
            digit_3 = torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(int(str(self.df.iloc[idx]['TARIFFNUMBER'])[3])), value=1)
        except:
            digit_3 = torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(0), value=1)
        digit_3 = digit_3[None,:]
        try:
            digit_4 = torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(int(str(self.df.iloc[idx]['TARIFFNUMBER'])[4])), value=1)
        except:
            digit_4 = torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(0), value=1)
        digit_4 = digit_4[None,:]
        try:
            digit_5 = torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(int(str(self.df.iloc[idx]['TARIFFNUMBER'])[5])), value=1)
        except:
            digit_5 = torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(0), value=1)
        digit_5 = digit_5[None,:]
        try:
            digit_6 = torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(int(str(self.df.iloc[idx]['TARIFFNUMBER'])[6])), value=1)
        except:
            digit_6 = torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(0), value=1)
        digit_6 = digit_6[None,:]
        try:
            digit_7 = torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(int(str(self.df.iloc[idx]['TARIFFNUMBER'])[7])), value=1)
        except:
            digit_7 = torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(0), value=1)
        digit_7 = digit_7[None,:]
        try:
            digit_8 = torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(int(str(self.df.iloc[idx]['TARIFFNUMBER'])[8])), value=1)
        except:
            digit_8 = torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(0), value=1)
        digit_8 = digit_8[None,:]
        try:
            digit_9 = torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(int(str(self.df.iloc[idx]['TARIFFNUMBER'])[9])), value=1)
        except:
            digit_9 = torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(0), value=1)
        digit_9 = digit_9[None,:]
        try:
            label_0 = int(str(self.df.iloc[idx]['PRMRYTARNR'])[0])
        except:
            label_0 = 0
        try:
            label_1 = int(str(self.df.iloc[idx]['PRMRYTARNR'])[1])
        except:
            label_1 = 0
        try:
            label_2 = int(str(self.df.iloc[idx]['PRMRYTARNR'])[2])
        except:
            label_2 = 0
        try:
            label_3 = int(str(self.df.iloc[idx]['PRMRYTARNR'])[3])
        except:
            label_3 = 0
        try:
            label_4 = int(str(self.df.iloc[idx]['PRMRYTARNR'])[4])
        except:
            label_4 = 0
        try:
            label_5 = int(str(self.df.iloc[idx]['PRMRYTARNR'])[5])
        except:
            label_5 = 0
        try:
            label_6 = int(str(self.df.iloc[idx]['PRMRYTARNR'])[6])
        except:
            label_6 = 0
        try:
            label_7 = int(str(self.df.iloc[idx]['PRMRYTARNR'])[7])
        except:
            label_7 = 0
        try:
            label_8 = int(str(self.df.iloc[idx]['PRMRYTARNR'])[8])
        except:
            label_8 = 0
        try:
            label_9 = int(str(self.df.iloc[idx]['PRMRYTARNR'])[9])
        except:
            label_9 = 0
        if self.transform:
            data = torch.cat([self.transform(data_general), self.transform(data_detail), digit_0, digit_1, digit_2, digit_3, digit_4, digit_5, digit_6, digit_7, digit_8, digit_9], dim=1)
        if self.target_transform:
            label = self.target_transform(label)
        return data, label_0, label_1, label_2, label_3, label_4, label_5, label_6, label_7, label_8, label_9