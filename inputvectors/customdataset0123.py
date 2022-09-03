import os
import pandas as pd
from torch.utils.data import Dataset

class CustomDataset_0123(Dataset):
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
        data = str(self.df.iloc[idx]['SIV_LIT_DSC_TE']) + " of type " + str(self.df.iloc[idx]['GOODSDESCRPTN'])
        # data = self.df.iloc[idx]['SIV_LIT_DSC_TE']
        try:
            label = int(str(self.df.iloc[idx]['PRMRYTARNR'])[:4])
        except:
            label = 0
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)
        return data, label