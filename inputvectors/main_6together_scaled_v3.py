from customdataset_6separate_v2 import CustomDataset_6Separate
# from passthroughtransformer import PassThroughTransformer
from getsentenceembedding import GetSentenceEmbedding
import torch
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import Lambda
from neuralnet_embedding_6separate_v4 import NeuralNetwork
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
    for batch, (X, y_0, y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9) in enumerate(dataloader):
        # Compute prediction and loss
        pred_0, pred_1, pred_2, pred_3, pred_4, pred_5, pred_6, pred_7, pred_8, pred_9 = model(X)
        # pred = torch.stack([pred_0, pred_1, pred_2, pred_3, pred_4, pred_5], dim=1)
        # y = torch.stack([y_0, y_1, y_2, y_3, y_4, y_5], dim=1)

        loss = 10*loss_fn(pred_0, y_0) + 9*loss_fn(pred_1, y_1) + 8*loss_fn(pred_2, y_2) + 7*loss_fn(pred_3, y_3) + 6*loss_fn(pred_4, y_4) + 5*loss_fn(pred_5, y_5) + 4*loss_fn(pred_6, y_6) + 3*loss_fn(pred_7, y_7) + 2*loss_fn(pred_8, y_8) + loss_fn(pred_9, y_9)

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
    test_loss, correct_6, correct_10 = 0, 0, 0

    print(f"Test Error: \n")

    with torch.no_grad():
        for X, y_0, y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9 in dataloader:
            pred_0, pred_1, pred_2, pred_3, pred_4, pred_5, pred_6, pred_7, pred_8, pred_9 = model(X)
            predict_6 = torch.stack([pred_0, pred_1, pred_2, pred_3, pred_4, pred_5], dim=1)
            act_6 = torch.stack([y_0, y_1, y_2, y_3, y_4, y_5], dim=1)
            predict_10 = torch.stack([pred_0, pred_1, pred_2, pred_3, pred_4, pred_5, pred_6, pred_7, pred_8, pred_9], dim=1)
            act_10 = torch.stack([y_0, y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9], dim=1)
            test_loss += 10*loss_fn(pred_0, y_0).item() + 9*loss_fn(pred_1, y_1).item() + 8*loss_fn(pred_2, y_2).item() + 7*loss_fn(pred_3, y_3).item() + 6*loss_fn(pred_4, y_4).item() + 5*loss_fn(pred_5, y_5).item() + 4*loss_fn(pred_6, y_6).item() + 3*loss_fn(pred_7, y_7).item() + 2*loss_fn(pred_8, y_8).item() + loss_fn(pred_9, y_9).item()

            correct_6 += (torch.all(predict_6.argmax(2)==act_6,dim=1)).type(torch.float).sum().item()
            correct_10 += (torch.all(predict_10.argmax(2)==act_10,dim=1)).type(torch.float).sum().item()

    test_loss /= num_batches
    correct_6 /= size
    correct_10 /= size

    print(f" Accuracy (6 digit): {(100*correct_6):>0.1f}%\n Accuracy (10 digit): {(100*correct_10):>0.1f}%\n Avg loss: {test_loss:>8f} \n")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# model = NeuralNetwork().to(device)
model = torch.load('./inputvectors/samplemodel/model_6together_scaled_v4.pth')
model.load_state_dict(torch.load('./inputvectors/samplemodel/model_6together_scaled_v4_weights.pth'))
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
torch.save(model, './inputvectors/model_6together_scaled_v4.pth')
torch.save(model.state_dict(), './inputvectors/model_6together_scaled_v4_weights.pth')
print("Done!")