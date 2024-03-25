import os
import subprocess
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


def ddp_setup():
    init_process_group(backend="nccl")


class Trainer:
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 loss_fn: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 save_every: int,
                 snapshot_path: str):
        
        self.local_rank = int(os.environ["LOCAL_RANK"])  # set by torchrun
        self.global_rank = int(os.environ["RANK"])  # set by torchrun

        self.model = model.to(self.local_rank)
        self.model = DDP(self.model, device_ids=[self.local_rank])

        self.train_loader = train_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.save_every = save_every
        self.epochs_run = 0

        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)


        hostname = subprocess.run(
            "echo $(hostname)", shell=True, stdout=subprocess.PIPE
        ).stdout
        self.hostname = hostname.decode().strip()


    def _load_snapshot(self, snapshot_path):
        snapshot = torch.load(snapshot_path)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]

    def _batch_pass(self, inputs, targets):
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss: torch.Tensor = self.loss_fn(outputs, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        print(f"HOST {self.hostname} GPU {self.local_rank} EPOCH {epoch}")
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.local_rank), targets.to(self.local_rank)
            self._batch_pass(inputs, targets)

    def _save_snapshot(self, epoch):
        snapshot = {}
        snapshot["MODEL_STATE"] = self.model.module.state_dict()
        snapshot["EPOCHS_RUN"] = epoch
        torch.save(snapshot, "snapshot.pt")
        print(f"EPOCH {epoch} checkpoint saved at snapshot.pt")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs + 1):
            self._run_epoch(epoch)
            if self.local_rank == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)


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


def main(total_epochs, save_every, snapshot_path: str = "snapshot.pt"):
    ddp_setup()
    train_dataset, model, loss_fn, optimizer = load_train_objects()
    train_loader = format_dataloader(train_dataset, 32)
    trainer = Trainer(
        model=model, 
        train_loader=train_loader, 
        loss_fn=loss_fn, 
        optimizer=optimizer, 
        save_every=save_every,
        snapshot_path=snapshot_path
    )
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    total_epochs = 50
    save_every = 1
    main(total_epochs, save_every)
