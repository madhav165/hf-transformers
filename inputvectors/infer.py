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
model = torch.load('./inputvectors/model_6together_scaled.pth')
model.load_state_dict(torch.load('./inputvectors/model_6together_scaled_weights.pth'))

# test_features, test_labels_0, test_labels_1, test_labels_2, test_labels_3, test_labels_4, test_labels_5 = next(iter(test_dataloader))
# pred_0, pred_1, pred_2, pred_3, pred_4, pred_5 = model(test_features)
# pred = torch.stack([pred_0, pred_1, pred_2, pred_3, pred_4, pred_5], dim=1)
# act = torch.stack([test_labels_0, test_labels_1, test_labels_2, test_labels_3, test_labels_4, test_labels_5], dim=1)
# print((torch.all(pred.argmax(2)==act,dim=1)).type(torch.float).sum().item())

size = len(test_dataloader.dataset)
num_batches = len(test_dataloader)
correct_6, correct_4, correct_2 = 0, 0, 0

with torch.no_grad():
    for test_features, y_0, y_1, y_2, y_3, y_4, y_5 in test_dataloader:
        pred_0, pred_1, pred_2, pred_3, pred_4, pred_5 = model(test_features)
        pred_6 = torch.stack([pred_0, pred_1, pred_2, pred_3, pred_4, pred_5], dim=1)
        act_6 = torch.stack([y_0, y_1, y_2, y_3, y_4, y_5], dim=1)
        pred_4 = torch.stack([pred_0, pred_1, pred_2, pred_3], dim=1)
        act_4 = torch.stack([y_0, y_1, y_2, y_3], dim=1)
        pred_2 = torch.stack([pred_0, pred_1], dim=1)
        act_2 = torch.stack([y_0, y_1], dim=1)

        correct_6 += (torch.all(pred_6.argmax(2)==act_6,dim=1)).type(torch.float).sum().item()
        correct_4 += (torch.all(pred_4.argmax(2)==act_4,dim=1)).type(torch.float).sum().item()
        correct_2 += (torch.all(pred_2.argmax(2)==act_2,dim=1)).type(torch.float).sum().item()

correct_6 /= size
correct_4 /= size
correct_2 /= size

print(f" Accuracy (6 digit): {(100*correct_6):>0.1f}%")
print(f" Accuracy (4 digit): {(100*correct_4):>0.1f}%")
print(f" Accuracy (2 digit): {(100*correct_2):>0.1f}%")