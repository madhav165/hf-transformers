import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(768, 4096),
            nn.ReLU(),
            nn.Linear(4096, 512),
        )
        self.linear_relu_digit_0 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(522, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
        self.linear_relu_digit_1 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(522, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
        self.linear_relu_digit_2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(522, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
        self.linear_relu_digit_3 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(522, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
        self.linear_relu_digit_4 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(522, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
        self.linear_relu_digit_5 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(522, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
        self.linear_relu_digit_6 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(522, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
        self.linear_relu_digit_7 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(522, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
        self.linear_relu_digit_8 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(522, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
        self.linear_relu_digit_9 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(522, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits_0 = self.linear_relu_digit_0(torch.cat([self.linear_relu_stack(x[:,:768]), x[:,768:768+10]], dim=1))
        logits_1 = self.linear_relu_digit_1(torch.cat([self.linear_relu_stack(x[:,:768]), x[:,768+10:768+20]], dim=1))
        logits_2 = self.linear_relu_digit_2(torch.cat([self.linear_relu_stack(x[:,:768]), x[:,768+20:768+30]], dim=1))
        logits_3 = self.linear_relu_digit_3(torch.cat([self.linear_relu_stack(x[:,:768]), x[:,768+30:768+40]], dim=1))
        logits_4 = self.linear_relu_digit_4(torch.cat([self.linear_relu_stack(x[:,:768]), x[:,768+40:768+50]], dim=1))
        logits_5 = self.linear_relu_digit_5(torch.cat([self.linear_relu_stack(x[:,:768]), x[:,768+50:768+60]], dim=1))
        logits_6 = self.linear_relu_digit_6(torch.cat([self.linear_relu_stack(x[:,:768]), x[:,768+60:768+70]], dim=1))
        logits_7 = self.linear_relu_digit_7(torch.cat([self.linear_relu_stack(x[:,:768]), x[:,768+70:768+80]], dim=1))
        logits_8 = self.linear_relu_digit_8(torch.cat([self.linear_relu_stack(x[:,:768]), x[:,768+80:768+90]], dim=1))
        logits_9 = self.linear_relu_digit_9(torch.cat([self.linear_relu_stack(x[:,:768]), x[:,768+90:768+100]], dim=1))
        return logits_0, logits_1, logits_2, logits_3, logits_4, logits_5, logits_6, logits_7, logits_8, logits_9