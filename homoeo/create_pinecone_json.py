import pandas as pd
import json

df = pd.read_csv('temple_details_embeddings_v33.csv')
num_pages = df.shape[0]
df0 = pd.read_csv('temple_details_v33.csv')
df0 = df0.drop_duplicates()
j_arr = []
for index, row in df.iterrows():
    d = {}
    d['id'] = f'item_{index}'
    d['score'] = 0
    d['values'] = row[3:].tolist()
    d2 = {}
    d2['pdf_numpages'] = num_pages
    d2['source'] = f"{row['state']} - {row['temple_name']} - {str(row['sent_grp'])}"
    print(row['state'], row['temple_name'])
    try:
        d2['text'] = df0.loc[(df0['state']==row['state'])&(df0['temple_name']==row['temple_name'])]['temple_details'].values[0]
    except:
        d2['text'] = ''
    d['metadata'] = d2
    j_arr.append(d)

j = {}
j['vectors'] = j_arr
j['namespace'] = 'india_vectors'

with open('india_vectors.json', 'w') as f:
    json.dump(j, f)