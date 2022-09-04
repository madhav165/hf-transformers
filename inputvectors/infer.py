from customdataset_6separate import CustomDataset_6Separate
# from passthroughtransformer import PassThroughTransformer
from getsentenceembedding import GetSentenceEmbedding
import torch
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import Lambda
from neuralnet_embedding_6separate import NeuralNetwork
from torch import nn
import numpy as np
import pandas as pd

torch.manual_seed(1)

dataset_0 = CustomDataset_6Separate('./inputvectors/nldata', transform=GetSentenceEmbedding())
train_size = int(0.8 * len(dataset_0))
test_size = len(dataset_0) - train_size
train_dataset, test_dataset = random_split(dataset_0, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True, drop_last=True)
model = torch.load('./inputvectors/model_6separate.pth')
model.load_state_dict(torch.load('./inputvectors/model_6separate_weights.pth'))

# test_features, test_labels_0, test_labels_1, test_labels_2, test_labels_3, test_labels_4, test_labels_5 = next(iter(test_dataloader))
# pred_0, pred_1, pred_2, pred_3, pred_4, pred_5 = model(test_features)
# pred = torch.stack([pred_0, pred_1, pred_2, pred_3, pred_4, pred_5], dim=1)
# act = torch.stack([test_labels_0, test_labels_1, test_labels_2, test_labels_3, test_labels_4, test_labels_5], dim=1)
# print(torch.all(pred.argmax(2)==act,dim=1))

size = len(test_dataloader.dataset)
num_batches = len(test_dataloader)

pred_all = torch.Tensor()
act_all = torch.Tensor()

with torch.no_grad():
    for test_features, test_labels_0, test_labels_1, test_labels_2, test_labels_3, test_labels_4, test_labels_5 in test_dataloader:
        pred_0, pred_1, pred_2, pred_3, pred_4, pred_5 = model(test_features)
        pred_0_argmax = pred_0.argmax(1)
        pred_1_argmax = pred_1.argmax(1)
        pred_2_argmax = pred_2.argmax(1)
        pred_3_argmax = pred_3.argmax(1)
        pred_4_argmax = pred_4.argmax(1)
        pred_5_argmax = pred_5.argmax(1)
        pred = torch.stack([pred_0_argmax, pred_1_argmax, pred_2_argmax, pred_3_argmax, pred_4_argmax, pred_5_argmax], dim=1)
        act = torch.stack([test_labels_0, test_labels_1, test_labels_2, test_labels_3, test_labels_4, test_labels_5], dim=1)
        pred_all = torch.cat([pred_all, pred], dim=0)
        act_all = torch.cat([act_all, act], dim=0)
        print(pred_all.size())

pred_all = pd.DataFrame(pred_all.numpy())
act_all = pd.DataFrame(act_all.numpy())
pred_all['All'] = pred_all[0].astype(str)+pred_all[1].astype(str)+pred_all[2].astype(str)+pred_all[3].astype(str)+pred_all[4].astype(str)+pred_all[5].astype(str)
pred_all['Chapter'] = pred_all[0].astype(str)+pred_all[1].astype(str)
pred_all['Four digit'] = pred_all[0].astype(str)+pred_all[1].astype(str)+pred_all[2].astype(str)+pred_all[3].astype(str)
act_all['All'] = act_all[0].astype(str)+act_all[1].astype(str)+act_all[2].astype(str)+act_all[3].astype(str)+act_all[4].astype(str)+act_all[5].astype(str)
act_all['Chapter'] = act_all[0].astype(str)+act_all[1].astype(str)
act_all['Four digit'] = act_all[0].astype(str)+act_all[1].astype(str)+act_all[2].astype(str)+act_all[3].astype(str)

print('All accuracy = {0:.2%}'.format((pred_all['All']==act_all['All']).sum()/len(pred_all)))
print('Chapter accuracy = {0:.2%}'.format((pred_all['Chapter']==act_all['Chapter']).sum()/len(pred_all)))
print('Four digit accuracy = {0:.2%}'.format((pred_all['Four digit']==act_all['Four digit']).sum()/len(pred_all)))

pred_all.to_csv('./inputvectors/pred_all.csv', index=False)
act_all.to_csv('./inputvectors/act_all.csv', index=False)