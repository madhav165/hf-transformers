import pandas as pd
import os
from dotenv import load_dotenv
import numpy as np
import openai
import pickle
import tiktoken
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    wait_random
)  # for exponential backoff

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

COMPLETIONS_MODEL = "text-davinci-003"
EMBEDDING_MODEL = "text-embedding-ada-002"

@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(20))
def get_embedding(text: str, model: str=EMBEDDING_MODEL) -> list[float]:
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]

def compute_doc_embeddings(df: pd.DataFrame) -> dict[tuple[str, str], list[float]]:
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
    
    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    return {
        idx: get_embedding(r.content) for idx, r in df.iterrows()
    }

df = pd.read_csv('clarke_symptoms_cleaned_v2.csv')
df['content'] = df['content'].str.replace(' l. ', ' left ')
df['content'] = df['content'].str.replace(' r. ', ' right ')
df['content'] = df['content'].str.replace(' l.;', ' left;')
df['content'] = df['content'].str.replace(' r.;', ' right;')
headings = df['heading'].unique().tolist()
for h in headings:
    print(h)
    df.loc[df['heading']==h].to_csv(f'parts/clarke_symptoms_cleaned_v2_{h}.csv', index=False)
    document_embeddings = compute_doc_embeddings(df.loc[df['heading']==h])
    pd.DataFrame(document_embeddings.items()).to_csv(f'clarke_symptoms_cleaned_v2_{h}_embeddings.csv', index=False)