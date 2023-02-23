import pandas as pd
from ast import literal_eval

df = pd.read_csv('clarke_remedy_info_cleaned_v2_embeddings.csv')

df2 = df['0'].apply(lambda x: pd.Series([literal_eval(x)[0], literal_eval(x)[1]]))
df2.columns=['title', 'heading']

df3 = df['1'].apply(lambda x: pd.Series(literal_eval(x)))

df2 = pd.concat([df2, df3], axis=1)
df2.to_csv('clarke_remedy_info_cleaned_v2_embeddings_v2.csv', index=False)