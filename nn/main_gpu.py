from torch.utils.data import random_split, DataLoader
import torch
from torch import nn
from datetime import datetime
import logging
from customdataset_gpu import CustomDataset
from getsentenceembedding_gpu import GetSentenceEmbedding
from neuralnet_gpu import NeuralNetwork

torch.manual_seed(1)
# torch.multiprocessing.set_start_method('spawn', force=True)

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

def train_loop(dataloader, device, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y_0, y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9) in enumerate(dataloader):
        # Compute prediction and loss
        pred_0, pred_1, pred_2, pred_3, pred_4, pred_5, pred_6, pred_7, pred_8, pred_9 = model(X)
        
        loss = 10*loss_fn(pred_0, y_0[:,0]) + 9*loss_fn(pred_1, y_1[:,0]) + 8*loss_fn(pred_2, y_2[:,0]) + 7*loss_fn(pred_3, y_3[:,0]) \
        + 6*loss_fn(pred_4, y_4[:,0]) + 5*loss_fn(pred_5, y_5[:,0]) + 4*loss_fn(pred_6, y_6[:,0]) + 3*loss_fn(pred_7, y_7[:,0]) + 2*loss_fn(pred_8, y_8[:,0]) \
        + loss_fn(pred_9, y_9[:,0])

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 20 == 0:
            loss, current = loss.item(), (batch+1) * len(X)
            logging.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, device, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct_6, correct_10 = 0, 0, 0

    logging.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Test Error: \n")

    with torch.no_grad():
        for batch, (X, y_0, y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9) in enumerate(dataloader):
            pred_0, pred_1, pred_2, pred_3, pred_4, pred_5, pred_6, pred_7, pred_8, pred_9 = model(X)
            predict_6 = torch.stack([pred_0, pred_1, pred_2, pred_3, pred_4, pred_5], dim=1)
            act_6 = torch.stack([y_0[:,0], y_1[:,0], y_2[:,0], y_3[:,0], y_4[:,0], y_5[:,0]], dim=1)
            predict_10 = torch.stack([pred_0, pred_1, pred_2, pred_3, pred_4, pred_5, pred_6, pred_7, pred_8, pred_9], dim=1)
            act_10 = torch.stack([y_0[:,0], y_1[:,0], y_2[:,0], y_3[:,0], y_4[:,0], y_5[:,0], y_6[:,0], y_7[:,0], y_8[:,0], y_9[:,0]], dim=1)
            test_loss += 10*loss_fn(pred_0, y_0[:,0]).item() + 9*loss_fn(pred_1, y_1[:,0]).item() + 8*loss_fn(pred_2, y_2[:,0]).item() \
            + 7*loss_fn(pred_3, y_3[:,0]).item() + 6*loss_fn(pred_4, y_4[:,0]).item() + 5*loss_fn(pred_5, y_5[:,0]).item() + 4*loss_fn(pred_6, y_6[:,0]).item() \
            + 3*loss_fn(pred_7, y_7[:,0]).item() + 2*loss_fn(pred_8, y_8[:,0]).item() + loss_fn(pred_9, y_9[:,0]).item()
            
            correct_6 += (torch.all(predict_6.argmax(2)==act_6,dim=1)).type(torch.float).sum().item()
            correct_10 += (torch.all(predict_10.argmax(2)==act_10,dim=1)).type(torch.float).sum().item()
            
            if (batch % 20 == 0):
                logging.info(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}:")
                logging.info(f'6 digit accuracy: {(100*correct_6/((batch+1) * len(X))):>0.1f}%, 10 digit accuracy: {(100*correct_10/((batch+1) * len(X))):>0.1f}% [{(batch+1) * len(X):>5d}/{size:>5d}]')

    test_loss /= num_batches
    correct_6 /= size
    correct_10 /= size

    logging.info(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}:")
    logging.info(f"\n Validation accuracy (6 digit): {(100*correct_6):>0.1f}%")
    logging.info(f"\n Validation accuracy (10 digit): {(100*correct_10):>0.1f}%")
    logging.info(f"\n Avg validation loss: {test_loss:>8f} \n")

flist=[
'./custom-nn/nldata/dr-nl-import-training-2022-05-31_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-06-01_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-06-02_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-06-03_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-06-04_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-06-05_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-06-06_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-06-07_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-06-08_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-06-09_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-06-11_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-06-12_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-06-13_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-06-14_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-06-15_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-06-16_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-06-17_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-06-18_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-06-19_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-06-20_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-06-21_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-06-22_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-06-23_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-06-24_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-06-25_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-06-26_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-06-27_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-06-28_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-06-29_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-06-30_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-07-01_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-07-02_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-07-03_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-07-04_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-07-07_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-07-08_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-07-09_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-07-10_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-07-11_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-07-12_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-07-13_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-07-14_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-07-15_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-07-16_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-07-17_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-07-18_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-07-19_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-07-20_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-07-21_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-07-22_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-07-23_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-07-24_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-07-25_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-07-26_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-07-27_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-07-28_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-07-29_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-07-30_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-07-31_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-08-01_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-08-02_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-08-03_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-08-04_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-08-05_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-08-07_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-08-08_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-08-09_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-08-10_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-08-11_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-08-13_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-08-14_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-08-15_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-08-16_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-08-17_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-08-18_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-08-19_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-08-20_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-08-21_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-08-22_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-08-23_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-08-24_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-08-25_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-08-26_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-09-01_processed.csv'
,'./custom-nn/nldata/dr-nl-import-training-2022-09-02_processed.csv'
]

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
logging.info(f"Using {device} device")

dataset_0 = CustomDataset(flist, device, transform=GetSentenceEmbedding(device))
train_size = int(0.7 * len(dataset_0))
test_size = int(0.15 * len(dataset_0))
valid_size = len(dataset_0) - train_size - test_size
train_dataset, test_dataset, valid_dataset = random_split(dataset_0, [train_size, test_size, valid_size])

train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=True, drop_last=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=512, shuffle=True, drop_last=True)

model = NeuralNetwork().to(device)
logging.info(model)

learning_rate = 1e-2
epochs = 5
momentum=0.9

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

for t in range(epochs):
    logging.info(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(valid_dataloader, model, loss_fn)
    scheduler.step()

torch.save(model, './custom-nn/model_gpu_v1.pth')
torch.save(model.state_dict(), './custom-nn/model_gpu_v1_weights.pth')
print("Done!")