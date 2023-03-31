import pandas as pd
import re
from typing import Set
from transformers import GPT2TokenizerFast

import numpy as np
from nltk.tokenize import sent_tokenize

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def count_tokens(text: str) -> int:
    """count the number of tokens in a string"""
    return len(tokenizer.encode(text))

def reduce_long(
    long_text: str, long_text_tokens: bool = False, max_len: int = 590
) -> str:
    """
    Reduce a long text to a maximum of `max_len` tokens by potentially cutting at a sentence end
    """
    if not long_text_tokens:
        long_text_tokens = count_tokens(long_text)
    if long_text_tokens > max_len:
        sentences = sent_tokenize(long_text.replace("\n", " "))
        ntokens = 0
        for i, sentence in enumerate(sentences):
            ntokens += 1 + count_tokens(sentence)
            if ntokens > max_len:
                return ". ".join(sentences[:i][:-1]) + "."

    return long_text



# df = pd.read_csv('clarke_remedy_info.csv', encoding = 'ISO-8859-1')

# df2 = df.set_index(['Name', 'URL']).stack().reset_index()
# df2.drop(columns='URL', inplace=True)

# df2.columns=['title', 'heading', 'content']
# df2['content'] = df2['content'].apply(lambda x: reduce_long(x, True))
# df2['tokens'] = df2['content_reduced'].apply(lambda x: count_tokens(x))
# df2.to_csv('clarke_remedy_info_cleaned.csv', index=False)


# df = pd.read_csv('clarke_symptoms.csv', encoding = 'ISO-8859-1')
# df.drop(columns='URL', inplace=True)
# df.columns=['title', 'heading', 'content']
# df['content'] = df['content'].apply(lambda x: reduce_long(x, max_len=1024))
# df['tokens'] = df['content'].apply(lambda x: count_tokens(x))
# df.to_csv('clarke_symptoms_cleaned.csv', index=False)

# for homoeo
# df = pd.read_csv('clarke_symptoms_cleaned_v2.csv', encoding = 'ISO-8859-1')
# df.columns=['title', 'heading', 'content']
# df['content'] = df['content'].apply(lambda x: reduce_long(x, max_len=1024))
# df['tokens'] = df['content'].apply(lambda x: count_tokens(x))
# df.to_csv('clarke_symptoms_cleaned_v2.csv', index=False)

# for temple site
max_len = 300
df = pd.read_csv('temple_details.csv', encoding = 'ISO-8859-1')
df['state'] = df['state'].apply(lambda x: str(x).replace(',','_').replace(' ', '_'))
df['temple_name'] = df['temple_name'].apply(lambda x: str(x).replace(',','_').replace(' ', '_'))
df = df[['state', 'temple_name', 'temple_details']]
df.shape[0] #2441

df2 = df['temple_details'].apply(lambda x: sent_tokenize(x.replace("\n", " "))).apply(pd.Series).stack().reset_index()
df2 = df2.rename(columns={0: 'text'})
df2['tokens'] = df2['text'].apply(count_tokens)
df2['tot_tokens'] = df2.groupby(['level_0'])['tokens'].cumsum()
df2['tot_grp'] = df2['tot_tokens'].apply(lambda x: x//max_len)
df2 = df2.groupby(['level_0', 'tot_grp'])['text'].apply(' '.join).reset_index()
df = df.reset_index()
df2 = df2.merge(df[['index', 'state', 'temple_name']], left_on='level_0', right_on='index', how='left')
df2 = df2[['state', 'temple_name', 'tot_grp', 'text']]
df2 = df2.rename(columns={'tot_grp': 'sent_grp', 'text': 'temple_details'})
df2['tokens'] = df2['temple_details'].apply(count_tokens)

df2.to_csv('temple_details_v32.csv', index=False)

df3 = pd.read_csv('temple_details_v32.csv')
df3['tokens'] = df3['temple_details'].apply(count_tokens)
df3.to_csv('temple_details_v33.csv', index=False)