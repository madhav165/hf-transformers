import pandas as pd
import os
from torch import nn
import numpy as np
os.environ['TRANSFORMERS_CACHE'] = os.path.join(os.getcwd(), str('/'.join(['models','pretrained'])))
os.environ['HF_HOME'] = os.path.join(os.getcwd(), str('/'.join(['datasets','prebuilt'])))

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_embedding(s):
    # Sentences we want sentence embeddings for
    sentences = [s]

    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    # print("Sentence embeddings:")
    # print(sentence_embeddings)

    return sentence_embeddings[0]

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

df_train = pd.read_csv(os.path.join('sentencesimilarity', 'train.csv'))
parent_codes = df_train.loc[df_train['Parent'].isna()]['Code.1']
chapter_codes = df_train.loc[df_train['Parent.1'].isin(parent_codes)]['Code.1']
major_codes = df_train.loc[df_train['Parent.1'].isin(chapter_codes)]['Code.1']
df_train = df_train.loc[~((df_train['Code.1'].isin(parent_codes))|(df_train['Code.1'].isin(chapter_codes))|(df_train['Code.1'].isin(major_codes)))]

df_nl = pd.read_csv(os.path.join('sentencesimilarity', 'nl_data.csv'))
df_nl = df_nl.fillna('')
pd.options.display.float_format = '{:.2f}'.format

s1_list = (df_nl.iloc[:,2].apply(lambda x: str.lower(x)) + " " + df_nl.iloc[:,2].apply(lambda x: str.lower(x)) + " " + df_nl.iloc[:,6]).values.tolist()
s2_list = (df_train['Self-explanatory texts'] + " " + df_train['Code.1']).values.tolist()

s1_embedding_list =  []
for i, s1 in enumerate(s1_list):
    if i % 50 == 0:
        print(i)
    e = get_embedding(s1)
    s1_embedding_list.append(e)

s2_embedding_list =  []
for i, s2 in enumerate(s2_list):
    if i % 50 == 0:
        print(i)
    e = get_embedding(s2)
    s2_embedding_list.append(e)

res_arr = []
for i, s1 in enumerate(s1_embedding_list):
    if i % 50 == 0:
        print(i)
    score_arr = []
    for s2 in s2_embedding_list:
        score = np.dot(s1, s2)
        score_arr.append(score)
    pred_index = np.argmax(score_arr)
    # max_2 = np.partition(np.array(score_arr).flatten(), -2)[-2]
    # pred_index = np.where(score_arr == max_2)[0][0]
    pred = df_train.iloc[pred_index]['Code.1'].replace(' ','')
    act_desc = df_nl.iloc[i]['SIV_LIT_DSC_TE']
    act = df_nl.iloc[i]['PRMRYTARNR'][:6]
    s1 = df_nl.iloc[i]['GOODSDESCRPTN']
    s2 = df_train.iloc[np.argmax(score_arr)]['Self-explanatory texts']
    res_arr.append([act_desc, s1, act, pred, s2])

df_res = pd.DataFrame(res_arr)
df_res.columns=['UPS SIV_LIT_DSC_TE', 'UPS GOODSDESCRPTN', 'Actual PRMRYTARNR', 'Predicted Code', 'HTS Self-explanatory texts']
df_res.loc[:,'Actual chapter'] = df_res['Actual PRMRYTARNR'].apply(lambda x: x[:2])
df_res.loc[:,'Predicted chapter'] = df_res['Predicted Code'].apply(lambda x: x[:2])
tariff_acc = (df_res.loc[df_res['Actual PRMRYTARNR']==df_res['Predicted Code']]).shape[0]/df_res.shape[0]
chapter_acc = (df_res.loc[df_res['Actual chapter']==df_res['Predicted chapter']]).shape[0]/df_res.shape[0]
print('Tariff accuracy: {}'.format(tariff_acc))
print('Chapter accuracy: {}'.format(chapter_acc))

df_res.to_csv('./sentencesimilarity/result.csv', index=False)