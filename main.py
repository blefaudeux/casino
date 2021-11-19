#!/usr/bin/env python
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from train import train_model, Strategy
from model import Model


def run(rank, size):
    """Distributed function to be implemented later."""
    train_model(
        rank,
        size,
        epochs=1,
        batch_size=32,
        cpu_per_process=4,
        model_name=Model.Resnet18,
        warmup=40,
        learning_rate=1e-2,
        pruning_ratio_growth=0.025,
        pruning_max_ratio=0.5,
        sync_interval=60,
        strategy=Strategy.PRUNE_SORT,
        hysteresis=50,
    )


def init_process(rank, size, fn, backend="gloo"):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 2
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)
