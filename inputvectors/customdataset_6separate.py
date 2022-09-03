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
        # data = self.df.iloc[idx]['SIV_LIT_DSC_TE']
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
        if self.transform:
            data = torch.cat([self.transform(data_general), self.transform(data_detail)], dim=1)
        if self.target_transform:
            label = self.target_transform(label)
        return data, label_0, label_1, label_2, label_3, label_4, label_5