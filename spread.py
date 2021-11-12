import torch
import torchvision
import torchvision.transforms as transforms
from enum import Enum, auto
import os


def spread_lottery_tickets(rank: int, size: int, model: torch.nn.Module, test_model):
    """
    One naive method: check how all the agents are doing,
    pick the weights from the best one
    """
    with torch.no_grad():
        # Test the models across all ranks, EMA with the best one
        print(rank, "Evaluating the model")
        acc = torch.tensor(test_model(rank, model, batch_size=16))

        # Fetch all the results
        acc_list = [acc.clone() for _ in range(size)]
        torch.distributed.all_gather(acc_list, acc)

        scores = torch.tensor([t.item() for t in acc_list])
        winner = torch.argmax(scores).item()

        print(rank, f"Rank {winner} is the winner. Syncing")

        # Pull the winner model in and continue
        for p in model.parameters():
            torch.distributed.broadcast(p.data, src=winner)

        print(rank, "Sync done\n")
