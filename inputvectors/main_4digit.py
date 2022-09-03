from customdataset0123 import CustomDataset_0123
# from passthroughtransformer import PassThroughTransformer
from getsentenceembedding import GetSentenceEmbedding
import torch
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import Lambda
from neuralnet_embedding_4digit import NeuralNetwork
from torch import nn

torch.manual_seed(1)

# dataset_0 = CustomDataset_0('./inputvectors/nldata', transform=GetSentenceEmbedding(),
# target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))
dataset_0 = CustomDataset_0123('./inputvectors/nldata', transform=GetSentenceEmbedding())
train_size = int(0.8 * len(dataset_0))
test_size = len(dataset_0) - train_size
train_dataset, test_dataset = random_split(dataset_0, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True, drop_last=True)

# train_features, train_labels = next(iter(train_dataloader))
# print(train_features[0])
# print(train_labels[0])

# # train_features = train_features.type(torch.FloatTensor)

# # print(f"Feature batch shape: {train_features.size()}")
# # print(f"Labels batch shape: {train_labels.size()}")
# # print(type(train_features[0]))
# # print(type(train_labels[0]))

# # X = train_features[0]
# # logits = model(X)
# # pred_probab = nn.Softmax(dim=1)(logits)
# # y_pred = pred_probab.argmax(1)
# # print(f"Predicted class: {y_pred}")



def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    correct2 = 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model = NeuralNetwork().to(device)
print(model)

learning_rate = 1e-2
batch_size = 64
epochs = 3
momentum=0.9

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.7)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
    scheduler.step()
torch.save(model.state_dict(), 'model_4digit_weights.pth')
print("Done!")