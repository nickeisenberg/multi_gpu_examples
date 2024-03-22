from collections.abc import Callable
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Dataset


def setup(rank, world_size):
    dist.init_process_group(
        backend='nccl',  # 'nccl' is recommended for GPU usage, 'gloo' for CPU
        init_method='env://',  # Initializes from environment variables
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.set_device(rank)


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


class RandomDataset(Dataset):
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
                     sampler: DistributedSampler,
                     loader: DataLoader,
                     criterion: Callable, 
                     optimizer: torch.optim.Optimizer,
                     logger: Logger):

    running_loss = 0.0
    sampler.set_epoch(epoch)
    for i, (imgs, labs) in enumerate(loader):
        if (i + 1) % 5 == 0:
            avg_loss = running_loss / (i + 1)
            hostname = os.getenv("HOSTNAME")
            logger.log(
                f"HOSTNAME {hostname} EPOCH {epoch} BATCH {i + 1} LOSS: {avg_loss}"
            )

        imgs, labs = imgs.cuda(), labs.cuda()

        loss = train_batch_pass(network, imgs, labs, criterion, optimizer)
        
        running_loss += loss.item()


def main():
    rank = int(os.getenv('SLURM_PROCID', '0'))
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    
    setup(rank, world_size)

    num_epochs = 2
    batch_size = 64

    net = Net().cuda()
    net = DDP(net, device_ids=[rank])

    train_dataset = RandomDataset()
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank
    )
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, sampler=train_sampler
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    logger = Logger("/g/g11/eisenbnt/projects/distributed_data_parallel/log.log")

    for epoch in range(num_epochs):
        train_epoch_pass(
            epoch=epoch, 
            network=net, 
            loader=train_loader,
            sampler=train_sampler,
            criterion=criterion, 
            optimizer=optimizer,
            logger=logger
        )


if __name__ == "__main__":
    rank = int(os.getenv('SLURM_PROCID', '0'))
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    hostname = os.getenv("HOSTNAME")
    print(rank)
    print(hostname)
    print(world_size)
    main()
