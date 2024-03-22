#! /g/g11/eisenbnt/venvs/base/bin/python3

import os
from collections.abc import Callable
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset as _Dataset
from torch.utils.data import DataLoader


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 64, 3, 1)
        self.c2 = nn.Conv2d(64, 128, 3, 1)
        self.c3 = nn.Conv2d(128, 256, 3, 2)
        self.linear = nn.Linear(952576, 10)

    def forward(self, x):
        x = torch.relu(self.c1(x))
        x = torch.relu(self.c2(x))
        x = torch.relu(self.c3(x))
        output = self.linear(nn.Flatten()(x))
        return output


class Dataset(_Dataset):
    def __init__(self):
        self.num_samples = 8000
        self.input_size = (3, 128, 128)
        self.output_classes = 10

    def __len__(self):
        return self.num_samples

    def __getitem__(self, _):
        return torch.randn(self.input_size), torch.randint(0, self.output_classes, (1,))[0]


class Logger:
    def __init__(self, save_to):
        self.save_to = save_to

    def log(self, loss):
        with open(self.save_to, "a") as wl:
            wl.write(loss + "\n")


def train_batch_pass(network: nn.Module,
                     inputs: torch.Tensor,
                     targets: torch.Tensor,
                     criterion: Callable,
                     optimizer: torch.optim.Optimizer):
    optimizer.zero_grad()
    outputs = network(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    return loss


def train_epoch_pass(epoch: int,
                     network: nn.Module,
                     loader: DataLoader,
                     device: str,
                     criterion: Callable,
                     optimizer: torch.optim.Optimizer,
                     logger: Logger):

    running_loss = 0.0

    for i, (imgs, labs) in enumerate(loader):
        if (i + 1) % 5 == 0:
            avg_loss = running_loss / (i + 1)
            logger.log(f"EPOCH {epoch} BATCH {i + 1} LOSS: {avg_loss}")
            hostname = os.getenv("HOSTNAME", "na")
            print(f"test from {hostname}")

        imgs, labs = imgs.to(device), labs.to(device)

        loss = train_batch_pass(network, imgs, labs, criterion, optimizer)

        running_loss += loss.item()


def main():
    num_epochs = 2
    batch_size = 256
    dataset = Dataset()
    dataloader = DataLoader(dataset, batch_size)
    device = "cuda:0"
    device_ids = [0, 1]
    net = Net()
    net = nn.DataParallel(net, device_ids=device_ids)
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    logger = Logger(
        "/g/g11/eisenbnt/projects/data_parallel/logger.log"
    )

    for epoch in range(num_epochs):
        train_epoch_pass(
            epoch=epoch,
            network=net,
            loader=dataloader,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            logger=logger
        )


if __name__ == "__main__":
    main()
