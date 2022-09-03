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
        loss_0 = loss_fn(pred_0, y_0)
        loss_1 = loss_fn(pred_1, y_1)
        loss_2 = loss_fn(pred_2, y_2)
        loss_3 = loss_fn(pred_3, y_3)
        loss_4 = loss_fn(pred_4, y_4)
        loss_5 = loss_fn(pred_5, y_5)

        # Backpropagation
        optimizer.zero_grad()
        loss_0.backward()
        loss_1.backward()
        loss_2.backward()
        loss_3.backward()
        loss_4.backward()
        loss_5.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss_0, current = loss_0.item(), batch * len(X)
            print(f"loss 0: {loss_0:>7f}  [{current:>5d}/{size:>5d}]")
            loss_1, current = loss_1.item(), batch * len(X)
            print(f"loss 1: {loss_1:>7f}  [{current:>5d}/{size:>5d}]")
            loss_2, current = loss_2.item(), batch * len(X)
            print(f"loss 2: {loss_2:>7f}  [{current:>5d}/{size:>5d}]")
            loss_3, current = loss_3.item(), batch * len(X)
            print(f"loss 3: {loss_3:>7f}  [{current:>5d}/{size:>5d}]")
            loss_4, current = loss_4.item(), batch * len(X)
            print(f"loss 4: {loss_4:>7f}  [{current:>5d}/{size:>5d}]")
            loss_5, current = loss_5.item(), batch * len(X)
            print(f"loss 5: {loss_5:>7f}  [{current:>5d}/{size:>5d}]")
            print("==========================")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss_0, test_loss_1, test_loss_2, test_loss_3, test_loss_4, test_loss_5 = 0, 0, 0, 0, 0, 0
    correct_0, correct_1, correct_2, correct_3, correct_4, correct_5 = 0, 0, 0, 0, 0, 0

    with torch.no_grad():
        for X, y_0, y_1, y_2, y_3, y_4, y_5 in dataloader:
            pred_0, pred_1, pred_2, pred_3, pred_4, pred_5 = model(X)
            test_loss_0 += loss_fn(pred_0, y_0).item()
            correct_0 += (pred_0.argmax(1) == y_0).type(torch.float).sum().item()
            test_loss_1 += loss_fn(pred_1, y_1).item()
            correct_1 += (pred_1.argmax(1) == y_1).type(torch.float).sum().item()
            test_loss_2 += loss_fn(pred_2, y_2).item()
            correct_2 += (pred_2.argmax(1) == y_2).type(torch.float).sum().item()
            test_loss_3 += loss_fn(pred_3, y_3).item()
            correct_3 += (pred_3.argmax(1) == y_3).type(torch.float).sum().item()
            test_loss_4 += loss_fn(pred_4, y_4).item()
            correct_4 += (pred_4.argmax(1) == y_4).type(torch.float).sum().item()
            test_loss_5 += loss_fn(pred_5, y_5).item()
            correct_5 += (pred_5.argmax(1) == y_5).type(torch.float).sum().item()

    test_loss_0 /= num_batches
    correct_0 /= size
    test_loss_1 /= num_batches
    correct_1 /= size
    test_loss_2 /= num_batches
    correct_2 /= size
    test_loss_3 /= num_batches
    correct_3 /= size
    test_loss_4 /= num_batches
    correct_4 /= size
    test_loss_5 /= num_batches
    correct_5 /= size
    print(f"Test Error: \n")
    print(f" Accuracy 0: {(100*correct_0):>0.1f}%, Avg loss 0: {test_loss_0:>8f} \n")
    print(f" Accuracy 1: {(100*correct_1):>0.1f}%, Avg loss 1: {test_loss_1:>8f} \n")
    print(f" Accuracy 2: {(100*correct_2):>0.1f}%, Avg loss 2: {test_loss_2:>8f} \n")
    print(f" Accuracy 3: {(100*correct_3):>0.1f}%, Avg loss 3: {test_loss_3:>8f} \n")
    print(f" Accuracy 4: {(100*correct_4):>0.1f}%, Avg loss 4: {test_loss_4:>8f} \n")
    print(f" Accuracy 5: {(100*correct_5):>0.1f}%, Avg loss 5: {test_loss_5:>8f} \n")

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
torch.save(model.state_dict(), 'model_6separate_weights.pth')
print("Done!")