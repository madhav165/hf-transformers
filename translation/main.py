import pandas as pd
import os
from torch import nn
os.environ['TRANSFORMERS_CACHE'] = os.path.join(os.getcwd(), str('/'.join(['models','pretrained'])))
os.environ['HF_HOME'] = os.path.join(os.getcwd(), str('/'.join(['datasets','prebuilt'])))

from transformers import pipeline

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")

flist = []
for (dirpath, dirnames, filenames) in os.walk(os.path.join('translation', 'data')):
    flist.extend(filenames)
    break
df = pd.DataFrame()
for fl in flist:
    df_fl = pd.read_csv(os.path.join('translation', 'data', fl))
    df = pd.concat([df, df_fl], axis=0)

text = df['SIV_LIT_DSC_TE'].values.tolist()
text = text[:150]
texts = pd.DataFrame(text)

# language_identifier=pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")
# res = language_identifier(text)
# langs = pd.DataFrame(res)

res = translator(text)
trans = pd.DataFrame(res)

# res = pd.concat([texts, langs, trans], axis=1)
res = pd.concat([texts, trans], axis=1)
res.to_csv('./translation/res2.csv')