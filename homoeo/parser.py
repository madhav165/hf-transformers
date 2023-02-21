import requests
from bs4 import BeautifulSoup
import re

url = 'http://www.homeoint.org/clarke/a/ars.htm'

resp = requests.get(url)
soup = BeautifulSoup(resp.content)
text = soup.text
text = re.sub(' +', ' ', text)
text = re.sub('\n ', ' ', text)

# text = re.sub('\n\n', '\t', text)
# text = re.sub('\n', ' ', text)
# text = re.sub('\t', '\n', text)
clinical = re.search('Clinical\.─((.|\n)*)(?=Characteristics\.─)', text).group(0)
clinical = re.sub('\n', ' ', clinical).strip()
characteristics = re.search('Characteristics\.─((.|\n)*)(?=Relations\.─)', text).group(0)
characteristics = re.sub('\n', ' ', characteristics).strip()
relations = re.search('Relations\.─((.|\n)*)(?=Causation\.─)', text).group(0)
relations = re.sub('\n', ' ', relations).strip()
causation = re.search('Causation\.─((.|\n)*)(?=SYMPTOMS\.)', text).group(0)
causation = re.sub('\n', ' ', causation).strip()

# with open ('ars.txt', 'w', encoding='utf-8') as f:
#     f.write(text)