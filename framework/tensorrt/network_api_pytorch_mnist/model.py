import os
import sys

# This file contains functions for training a PyTorch MNIST Model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import numpy as np

from random import randint

if os.name == 'nt':  # For Windows
    mnist_data_path = "D:\\park\\mnist_data"
elif os.name == 'posix' and sys.platform == 'darwin':  # For MacOS
    mnist_data_path = "/opt/park/mnist_data"
else:  # For Unix/Linux
    mnist_data_path = "/mnt/d/park/mnist_data"


# Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.fc1 = nn.Linear(800, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), kernel_size=2, stride=2)
        x = F.max_pool2d(self.conv2(x), kernel_size=2, stride=2)
        x = x.view(-1, 800)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class MnistModel(object):
    def __init__(self):
        self.batch_size = 64
        self.test_batch_size = 100
        self.learning_rate = 0.0025
        self.sgd_momentum = 0.9
        self.log_interval = 100

        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(mnist_data_path, train=True, download=True, transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]), ),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=1,
            timeout=600,
        )
        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                mnist_data_path,
                train=False,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
                ),
            ),
            batch_size=self.test_batch_size,
            shuffle=True,
            num_workers=1,
            timeout=600,
        )
        self.network = Net()
        if torch.cuda.is_available():
            self.network = self.network.to("cuda")

    # Train the network for one or more epochs, validating after each epoch.
    def learn(self, num_epochs=2):
        # Train the network for a single epoch
        def train(epoch):
            self.network.train()
            optimizer = optim.SGD(
                self.network.parameters(),
                lr=self.learning_rate,
                momentum=self.sgd_momentum,
            )
            for batch, (data, target) in enumerate(self.train_loader):
                if torch.cuda.is_available():
                    data = data.to("cuda")
                    target = target.to("cuda")
                data, target = Variable(data), Variable(target)
                optimizer.zero_grad()
                output = self.network(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
                if batch % self.log_interval == 0:
                    print(
                        "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                            epoch,
                            batch * len(data),
                            len(self.train_loader.dataset),
                            100.0 * batch / len(self.train_loader),
                            loss.data.item(),
                        )
                    )

        # Test the network
        def test(epoch):
            self.network.eval()
            test_loss = 0
            correct = 0
            for data, target in self.test_loader:
                with torch.no_grad():
                    if torch.cuda.is_available():
                        data = data.to("cuda")
                        target = target.to("cuda")
                    data, target = Variable(data), Variable(target)
                output = self.network(data)
                test_loss += F.nll_loss(output, target).data.item()
                pred = output.data.max(1)[1]
                correct += pred.eq(target.data).cpu().sum()
            test_loss /= len(self.test_loader)
            print(
                "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                    test_loss,
                    correct,
                    len(self.test_loader.dataset),
                    100.0 * correct / len(self.test_loader.dataset),
                )
            )

        for e in range(num_epochs):
            train(e + 1)
            test(e + 1)

    def get_weights(self):
        return self.network.state_dict()

    def get_random_testcase(self):
        data, target = next(iter(self.test_loader))
        case_num = randint(0, len(data) - 1)
        test_case = data.cpu().numpy()[case_num].ravel().astype(np.float32)
        test_name = target.cpu().numpy()[case_num]
        return test_case, test_name
