import torch.nn as nn
import torch.nn.functional as F
from opt_einsum.backends import torch
from torch import optim
import torch
import numpy as np
from torch.optim.lr_scheduler import StepLR


class Model(nn.Module):
    def __init__(self, in_feature, out_feature, dim=4048, embedding_dim=500):
        super().__init__()
        self.pool = nn.MaxPool2d((3, 1))

        self.conv1 = nn.Conv2d(in_channels=in_feature, out_channels=64, kernel_size=(3, 3), padding=(0, 2))
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), padding=(0, 2))
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), padding=(0, 2))
        self.bn3 = nn.BatchNorm2d(16)

        self.flat = nn.Flatten(1, 3)

        self.lin1 = nn.Linear(dim, embedding_dim)
        self.lin2 = nn.Linear(embedding_dim, out_feature)

        self.tanh = nn.Tanh()

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.pool(self.bn1(F.relu(x)))
        x = self.pool(self.bn2(F.relu(self.conv2(x))))
        x = self.pool(self.bn3(F.relu(self.conv3(x))))
        x = self.flat(x)
        x = F.relu(self.lin1(x))
        feature = x
        x = self.lin2(x)
        return x, feature


def train_and_eval(model, trainloader, testloader, device, lr=0.001, n_epoch=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)  # best 0.00001
    scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
    print('start training')
    for epoch in range(n_epoch):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].type(torch.LongTensor).to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 100 == 0:
                with torch.no_grad():
                    train_correct = 0
                    train_total = 0
                    for data in trainloader:
                        inputs, labels = data[0].to(device), data[1].type(torch.LongTensor).to(device)
                        labels = labels.int()
                        # calculate outputs by running images through the network
                        outputs, _ = model(inputs)
                        # the class with the highest energy is what we choose as prediction
                        _, predicted = torch.max(outputs.data, 1)
                        predicted = predicted.int()
                        train_total += labels.size(0)
                        train_correct += (predicted == labels).sum().item()

                    correct = 0
                    total = 0
                    for data in testloader:
                        inputs, labels = data[0].to(device), data[1].type(torch.LongTensor).to(device)
                        labels = labels.int()
                        # calculate outputs by running images through the network
                        outputs, _ = model(inputs)
                        # the class with the highest energy is what we choose as prediction
                        _, predicted = torch.max(outputs.data, 1)
                        predicted = predicted.int()
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                if i > 1:
                    print("Epoch: " + str(epoch) + " ||Iteration: " + str(i) + "|| loss: " + str(
                        running_loss / 100) + "|| Accuracy: " + str(train_correct / train_total) + "|| Val Accuracy: " + str(correct / total))
                running_loss = 0
        scheduler.step()

    print('Finished Training')
