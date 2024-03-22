#! /g/g11/eisenbnt/venvs/base/bin/python3

import torch.cuda as cuda
import time
from tqdm import tqdm


def tqdm_test():
    if not cuda.is_available():
        x="no cuda"
        pbar = tqdm(list(range(10)))
        for i in pbar:
            pbar.set_postfix(epoch=i, cuda=x)
            time.sleep(1)
    else:
        x="cuda"
        pbar = tqdm(list(range(10)))
        for i in pbar:
            pbar.set_postfix(epoch=i, cuda=x)
            time.sleep(1)


def write_to_log(text):
    log_path = "/g/g11/eisenbnt/projects/figure_out_llnl/log.log"
    with open(log_path, "a") as wf:
        wf.write(text + "\n")


def log_test():
    if not cuda.is_available():
        x="no cuda"
        for i in list(range(10)):
            time.sleep(1)
            write_to_log(f"{i}: {x}")
    else:
        num_devices = cuda.device_count()
        x="cuda"
        for i in list(range(10)):
            time.sleep(1)
            write_to_log(f"{i}: {x} device count {num_devices}")


def combo_test():
    if not cuda.is_available():
        x="no cuda"
        pbar = tqdm(list(range(10)))
        for i in pbar:
            pbar.set_postfix(epoch=i, cuda=x)
            write_to_log(f"{i}: {x}")
            time.sleep(1)
    else:
        num_devices = cuda.device_count()
        x="cuda"
        pbar = tqdm(list(range(10)))
        for i in pbar:
            pbar.set_postfix(epoch=i, cuda=x)
            write_to_log(f"{i}: {x} device count {num_devices}")
            time.sleep(1)


if __name__ == "__main__":
    combo_test()
