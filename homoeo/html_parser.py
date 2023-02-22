import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import logging

logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

remedy_df = pd.read_csv('clarke_remedy_info_old.csv')
urls = remedy_df['URL'].unique().tolist()
arr = []
for url in urls:
    log.info(url)
    resp = requests.get(url)
    soup = BeautifulSoup(resp.content, 'html.parser')
    text = soup.text
    text = re.sub(' +', ' ', text)
    text = re.sub('\n ', ' ', text)

    # text = re.sub('\n\n', '\t', text)
    # text = re.sub('\n', ' ', text)
    # text = re.sub('\t', '\n', text)
    try:
        clinical = re.search('Clinical\.─((.|\n)*)(?=Characteristics\.─)', text).group(0)
        clinical = re.sub('\n', ' ', clinical).strip()
        clinical = clinical.replace('Clinical.─', '')
    except Exception as e:
        log.error(e)
        clinical = ''
    if clinical == '':
        try:
            clinical = re.search('Clinical\.─((.|\n)*)(?=Relations\.─)', text).group(0)
            clinical = re.sub('\n', ' ', clinical).strip()
            clinical = clinical.replace('Clinical.─', '')
        except Exception as e:
            log.error(e)
            clinical = ''
    if clinical == '':
        try:
            clinical = re.search('Clinical\.─((.|\n)*)(?=Causation\.─)', text).group(0)
            clinical = re.sub('\n', ' ', clinical).strip()
            clinical = clinical.replace('Clinical.─', '')
        except Exception as e:
            log.error(e)
            clinical = ''
    if clinical == '':
        try:
            clinical = re.search('Clinical\.─((.|\n)*)(?=SYMPTOMS\.)', text).group(0)
            clinical = re.sub('\n', ' ', clinical).strip()
            clinical = clinical.replace('Clinical.─', '')
        except Exception as e:
            log.error(e)
            clinical = ''
    if clinical == '':
        try:
            clinical = re.search('Clinical\.─((.|\n)*)(?=Copyright © Médi-T ® 2000)', text).group(0)
            clinical = re.sub('\n', ' ', clinical).strip()
            clinical = clinical.replace('Clinical.─', '')
        except Exception as e:
            log.error(e)
            clinical = ''
    try:
        characteristics = re.search('Characteristics\.─((.|\n)*)(?=Relations\.─)', text).group(0)
        characteristics = re.sub('\n', ' ', characteristics).strip()
        characteristics = characteristics.replace('Characteristics.─', '')
    except Exception as e:
        log.error(e)
        characteristics = ''
    if characteristics == '':
        try:
            characteristics = re.search('Characteristics\.─((.|\n)*)(?=Causation\.─)', text).group(0)
            characteristics = re.sub('\n', ' ', characteristics).strip()
            characteristics = characteristics.replace('Characteristics.─', '')
        except Exception as e:
            log.error(e)
            characteristics = ''
    if characteristics == '':
        try:
            characteristics = re.search('Characteristics\.─((.|\n)*)(?=SYMPTOMS\.)', text).group(0)
            characteristics = re.sub('\n', ' ', characteristics).strip()
            characteristics = characteristics.replace('Characteristics.─', '')
        except Exception as e:
            log.error(e)
            characteristics = ''
    if characteristics == '':
        try:
            characteristics = re.search('Characteristics\.─((.|\n)*)(?=Copyright © Médi-T ® 2000)', text).group(0)
            characteristics = re.sub('\n', ' ', characteristics).strip()
            characteristics = characteristics.replace('Characteristics.─', '')
        except Exception as e:
            log.error(e)
            characteristics = ''
    try:
        relations = re.search('Relations\.─((.|\n)*)(?=Causation\.─)', text).group(0)
        relations = re.sub('\n', ' ', relations).strip()
        relations = relations.replace('Relations.─', '')
    except Exception as e:
        log.error(e)
        relations = ''
    if relations == '':
        try:
            relations = re.search('Relations\.─((.|\n)*)(?=SYMPTOMS\.)', text).group(0)
            relations = re.sub('\n', ' ', relations).strip()
            relations = relations.replace('Relations.─', '')
        except Exception as e:
            log.error(e)
            relations = ''
    if relations == '':
        try:
            relations = re.search('Relations\.─((.|\n)*)(?=Copyright © Médi-T ® 2000)', text).group(0)
            relations = re.sub('\n', ' ', relations).strip()
            relations = relations.replace('Relations.─', '')
        except Exception as e:
            log.error(e)
            relations = ''
    try:
        causation = re.search('Causation\.─((.|\n)*)(?=SYMPTOMS\.)', text).group(0)
        causation = re.sub('\n', ' ', causation).strip()
        causation = causation.replace('Causation.─', '')
    except Exception as e:
        log.error(e)
        causation = ''
    if causation == '':
        try:
            causation = re.search('Causation\.─((.|\n)*)(?=Copyright © Médi-T ® 2000)', text).group(0)
            causation = re.sub('\n', ' ', causation).strip()
            causation = causation.replace('Causation.─', '')
        except Exception as e:
            log.error(e)
            causation = ''
    arr.append([url, clinical, characteristics, relations, causation])

df_res = pd.DataFrame(arr)
df_res.columns=['URL', 'Clinical', 'Characteristics', 'Relations', 'Causation']
df_res.to_csv('df_res.csv', index=False)