import requests
from bs4 import BeautifulSoup
import re

url = 'http://www.homeoint.org/clarke/a/ars.htm'

resp = requests.get(url)
soup = BeautifulSoup(resp.content)
text = soup.text
text = re.sub(' +', ' ', text)
text = re.sub('\n ', ' ', text)
text = re.sub('\n\n', '\t', text)
text = re.sub('\n', ' ', text)
text = re.sub('\t', '\n', text)
with open ('ars.txt', 'w', encoding='utf-8') as f:
    f.write(text)