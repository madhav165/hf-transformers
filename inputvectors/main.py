import pandas as pd
import matplotlib.pyplot as plt
import os
from torch import nn
import torch
os.environ['TRANSFORMERS_CACHE'] = os.path.join(os.getcwd(), str('/'.join(['models','pretrained'])))
os.environ['HF_HOME'] = os.path.join(os.getcwd(), str('/'.join(['datasets','prebuilt'])))

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

flist = []
for (dirpath, dirnames, filenames) in os.walk('./inputvectors/nldata'):
    flist.extend(filenames)
    break
df = pd.DataFrame()
for fl in flist:
    df_fl = pd.read_csv(os.path.join('inputvectors', 'nldata', fl))
    df = pd.concat([df, df_fl], axis=0)


inp_texts = df['SIV_LIT_DSC_TE'].values.tolist()
out_codes = df['PRMRYTARNR'].values.tolist()
out_codes_0 = df['PRMRYTARNR'].apply(lambda x: int(str(x)[0])).values.tolist()
out_codes_1 = df['PRMRYTARNR'].apply(lambda x: int(str(x)[1])).values.tolist()
out_codes_2 = df['PRMRYTARNR'].apply(lambda x: int(str(x)[2])).values.tolist()
out_codes_3 = df['PRMRYTARNR'].apply(lambda x: int(str(x)[3])).values.tolist()
out_codes_4 = df['PRMRYTARNR'].apply(lambda x: int(str(x)[4])).values.tolist()
out_codes_5 = df['PRMRYTARNR'].apply(lambda x: int(str(x)[5])).values.tolist()

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token
inp_encoded = tokenizer(inp_texts[:10], padding=True, truncation=True, return_tensors='pt')
out_state = model(**inp_encoded)

result = out_state[0]

x = result[:,0]
y = torch.tensor(out_codes[:10])
y_0 = torch.tensor(out_codes_0[:10])
y_1 = torch.tensor(out_codes_1[:10])
y_2 = torch.tensor(out_codes_2[:10])
y_3 = torch.tensor(out_codes_3[:10])
y_4 = torch.tensor(out_codes_4[:10])
y_5 = torch.tensor(out_codes_5[:10])




# df_train = pd.read_csv(os.path.join('nldata', 'train.csv'))
# parent_codes = df_train.loc[df_train['Parent'].isna()]['Code.1']
# chapter_codes = df_train.loc[df_train['Parent.1'].isin(parent_codes)]['Code.1']
# major_codes = df_train.loc[df_train['Parent.1'].isin(chapter_codes)]['Code.1']
# df_train = df_train.loc[~((df_train['Code.1'].isin(parent_codes))|(df_train['Code.1'].isin(chapter_codes))|(df_train['Code.1'].isin(major_codes)))]

# y_train = df_train['Code.1'].apply(lambda x: str(x).replace(' ',''))
# x_train = df_train['Self-explanatory texts']
# tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
# tokenized_x_train = tokenizer([' '.join(x) for x in x_train], truncation=True)
# tokenized_y_train = tokenizer([' '.join(x) for x in y_train], truncation=True)
# tokenized_x_train['labels'] = tokenized_y_train['input_ids']
