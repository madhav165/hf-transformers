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
            nn.Linear(512, 64),
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
            nn.Linear(532, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
        self.linear_relu_digit_3 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(542, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
        self.linear_relu_digit_4 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(552, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
        self.linear_relu_digit_5 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(562, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits_0 = self.linear_relu_digit_0(self.linear_relu_stack(x))
        logits_1 = self.linear_relu_digit_1(torch.cat([self.linear_relu_stack(x), nn.Softmax(dim=1)(logits_0)], dim=1))
        logits_2 = self.linear_relu_digit_2(torch.cat([self.linear_relu_stack(x), nn.Softmax(dim=1)(logits_0), nn.Softmax(dim=1)(logits_1)], dim=1))
        logits_3 = self.linear_relu_digit_3(torch.cat([self.linear_relu_stack(x), nn.Softmax(dim=1)(logits_0), nn.Softmax(dim=1)(logits_1), nn.Softmax(dim=1)(logits_2)], dim=1))
        logits_4 = self.linear_relu_digit_4(torch.cat([self.linear_relu_stack(x), nn.Softmax(dim=1)(logits_0), nn.Softmax(dim=1)(logits_1), nn.Softmax(dim=1)(logits_2), nn.Softmax(dim=1)(logits_3)], dim=1))
        logits_5 = self.linear_relu_digit_5(torch.cat([self.linear_relu_stack(x), nn.Softmax(dim=1)(logits_0), nn.Softmax(dim=1)(logits_1), nn.Softmax(dim=1)(logits_2), nn.Softmax(dim=1)(logits_3), nn.Softmax(dim=1)(logits_4)], dim=1))
        return logits_0, logits_1, logits_2, logits_3, logits_4, logits_5