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

    # Create a tuple of (title, section_name, content, number of tokens)
    outputs += [(title, h, c, t) if t<max_len 
                else (title, h, reduce_long(c, max_len), count_tokens(reduce_long(c,max_len))) 
                    for h, c, t in zip(nheadings, ncontents, ncontent_ntokens)]
    
    return outputs

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

df = pd.read_csv('clarke_symptoms_cleaned_v2.csv', encoding = 'ISO-8859-1')
df.columns=['title', 'heading', 'content']
df['content'] = df['content'].apply(lambda x: reduce_long(x, max_len=1024))
df['tokens'] = df['content'].apply(lambda x: count_tokens(x))
df.to_csv('clarke_symptoms_cleaned_v2.csv', index=False)

