import os
import torch

rank = int(os.getenv("SLURM_PROCID", "0"))

device_id = int(rank % 2)

hostname = os.getenv("HOSTNAME", "NO HOST")


print(f"HOSTNAME {hostname} RANK {rank} DEVICE_ID {device_id}")

try:
    torch.cuda.set_device(device_id)
    print(f"CUDA device sucessfully set to cuda:{device_id}")
except:
    print(f"Cant set device to {device_id}. cuda:{device_id} does not exist")

try:
    x = torch.tensor([1, 2, 3]).cuda()
    print(f"Tensor sucessfully put to cuda:{device_id}")
except:
    print(f"Cant move to cuda. cuda:{device_id} does not exist")
