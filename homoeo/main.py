import pandas as pd
import torch
import os
import sys
import logging

logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

remedy_df = pd.read_csv('clarke_remedy_info.csv')
symptom_df = pd.read_csv('clarke_symptoms.csv')

remedy_df.pivot(index='Name', columns='Item', values='Description').to_csv('remedy_pivot.csv')

symptom_df.pivot(index='Name', columns='Area', values='Symptom').to_csv('symptom_pivot.csv')