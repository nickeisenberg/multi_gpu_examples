import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class Trainer:
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 loss_fn: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 gpu_id: int,
                 save_every: int):
        self.gpu_id = gpu_id
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.save_every = save_every

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
        ckp = self.model.state_dict()
        torch.save(ckp, "checkpoint.pt")
        print(f"EPOCH {epoch} CHECKPOINT SAVED")

    def train(self, max_epochs: int):
        for epoch in range(1, max_epochs + 1):
            self._run_epoch(epoch)
            if epoch % self.save_every == 0:
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
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def main(device, total_epochs, save_every):
    train_dataset, model, loss_fn, optimizer = load_train_objects()
    model = model.to(device)
    train_loader = format_dataloader(train_dataset, 32)
    trainer = Trainer(
        model=model, 
        train_loader=train_loader, 
        loss_fn=loss_fn, 
        optimizer=optimizer, 
        gpu_id=device, 
        save_every=save_every
    )
    trainer.train(total_epochs)


if __name__ == "__main__":
    total_epochs = 5
    save_every = 1
    device = 0
    main(device, total_epochs, save_every)
