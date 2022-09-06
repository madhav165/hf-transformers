from customdataset_6separate import CustomDataset_6Separate
# from passthroughtransformer import PassThroughTransformer
from getsentenceembedding import GetSentenceEmbedding
import torch
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import Lambda
from neuralnet_embedding_6separate_v3 import NeuralNetwork
from torch import nn

torch.manual_seed(1)

# dataset_0 = CustomDataset_0('./inputvectors/nldata', transform=GetSentenceEmbedding(),
# target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))
dataset_0 = CustomDataset_6Separate('./inputvectors/nldata', transform=GetSentenceEmbedding())
train_size = int(0.8 * len(dataset_0))
test_size = len(dataset_0) - train_size
train_dataset, test_dataset = random_split(dataset_0, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=True, drop_last=True)

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
    for batch, (X, y_0, y_1, y_2, y_3, y_4, y_5) in enumerate(dataloader):
        # Compute prediction and loss
        pred_0, pred_1, pred_2, pred_3, pred_4, pred_5 = model(X)
        # pred = torch.stack([pred_0, pred_1, pred_2, pred_3, pred_4, pred_5], dim=1)
        # y = torch.stack([y_0, y_1, y_2, y_3, y_4, y_5], dim=1)

        loss = loss_fn(pred_0, y_0) + loss_fn(pred_1, y_1) + loss_fn(pred_2, y_2) + loss_fn(pred_3, y_3) + loss_fn(pred_4, y_4) + loss_fn(pred_5, y_5)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    print(f"Test Error: \n")

    with torch.no_grad():
        for X, y_0, y_1, y_2, y_3, y_4, y_5 in dataloader:
            pred_0, pred_1, pred_2, pred_3, pred_4, pred_5 = model(X)
            pred = torch.stack([pred_0, pred_1, pred_2, pred_3, pred_4, pred_5], dim=1)
            y = torch.stack([y_0, y_1, y_2, y_3, y_4, y_5], dim=1)
            test_loss += 6*loss_fn(pred_0, y_0).item() + 5*loss_fn(pred_1, y_1).item() + 4*loss_fn(pred_2, y_2).item() + 3*loss_fn(pred_3, y_3).item() + 2*loss_fn(pred_4, y_4).item() + loss_fn(pred_5, y_5).item()

            correct += (torch.all(pred.argmax(2)==y,dim=1)).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    print(f" Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model = NeuralNetwork().to(device)
# model = torch.load('./inputvectors/samplemodel/model_6together_scaled_v2.pth')
# model.load_state_dict(torch.load('./inputvectors/samplemodel/model_6together_scaled_v2_weights.pth'))
print(model)

learning_rate = 3e-2
epochs = 7
momentum=0.9

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
    scheduler.step()
torch.save(model, './inputvectors/model_6together_scaled_v3.pth')
torch.save(model.state_dict(), './inputvectors/model_6together_scaled_v3_weights.pth')
print("Done!")