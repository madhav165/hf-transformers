from customdataset_6separate import CustomDataset_6Separate
# from passthroughtransformer import PassThroughTransformer
from getsentenceembedding import GetSentenceEmbedding
import torch
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import Lambda
from neuralnet_embedding_6separate import NeuralNetwork
from torch import nn

torch.manual_seed(1)

# dataset_0 = CustomDataset_0('./inputvectors/nldata', transform=GetSentenceEmbedding(),
# target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))
dataset_0 = CustomDataset_6Separate('./inputvectors/nldata', transform=GetSentenceEmbedding())
train_size = int(0.8 * len(dataset_0))
test_size = len(dataset_0) - train_size
train_dataset, test_dataset = random_split(dataset_0, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True, drop_last=True)
model = torch.load('model_6separate.pth')
model.load_state_dict(torch.load('model_6separate_weights.pth'))

test_features, test_labels_0, test_labels_1, test_labels_2, test_labels_3, test_labels_4, test_labels_5 = next(iter(test_dataloader))

print(model(test_features[0]))