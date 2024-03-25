import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group



def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


class Trainer:
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 loss_fn: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 gpu_id: int,
                 save_every: int):
        self.gpu_id = gpu_id
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.save_every = save_every
        self.model = DDP(model, device_ids=[self.gpu_id])

    def _batch_pass(self, inputs, targets):
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss: torch.Tensor = self.loss_fn(outputs, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        print(f"GPU {self.gpu_id} EPOCH {epoch}")
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.gpu_id), targets.to(self.gpu_id)
            self._batch_pass(inputs, targets)

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        torch.save(ckp, "checkpoint.pt")
        print(f"EPOCH {epoch} CHECKPOINT SAVED")

    def train(self, max_epochs: int):
        for epoch in range(1, max_epochs + 1):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)


class RandomDataset(Dataset):
    def __init__(self, len):
        self.len = len

    def __len__(self):
        return self.len

    def __getitem__(self, _):
        return torch.randn(10), torch.randint(0, 4, (1,)).squeeze(0)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 10)
        self.linear3 = nn.Linear(10, 5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        return x


def load_train_objects():
    train_dataset = RandomDataset(1000)
    model = Model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    return train_dataset, model, loss_fn, optimizer


def format_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )


def main(rank, world_size, total_epochs, save_every):
    ddp_setup(rank, world_size)
    train_dataset, model, loss_fn, optimizer = load_train_objects()
    model = model.to(rank)
    train_loader = format_dataloader(train_dataset, 32)
    trainer = Trainer(
        model=model, 
        train_loader=train_loader, 
        loss_fn=loss_fn, 
        optimizer=optimizer, 
        gpu_id=rank, 
        save_every=save_every
    )
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    total_epochs = 5
    save_every = 1
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, total_epochs, save_every), nprocs=world_size)






