import torch
import torchvision
import torchvision.transforms as transforms
from enum import Enum, auto
from spread import spread_lottery_tickets
from prune import (
    sync_exchange_lottery_tickets,
    freeze_pruned_weights,
    rewind_model,
    sync_exchange_lottery_tickets_sorted,
)
import copy
from collections import namedtuple
from model import Model, get_model
import math
from store import get_client_store, get_server_store

_DATASET = torchvision.datasets.CIFAR10  # torchvision.datasets.CIFAR10

TrainSettings = namedtuple(
    "TrainSettings", ["LR", "pruning_eps", "pruning_max_ratio", "strategy"]
)


def test_model(rank: int, model: torch.nn.Module, batch_size: int, transform) -> float:
    testset = _DATASET(
        root="./data", train=False, download=rank == 0, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)

            # only check the top1
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print(rank, f"Accuracy of the network on the 10000 test images: {accuracy}")
    return accuracy


class Strategy(Enum):
    PRUNE_SORT = auto()
    PRUNE_THRESHOLD = auto()
    PRUNE_GROW = auto()
    SPREAD = auto()


def train_model(
    rank: int,
    world_size: int,
    epochs: int,
    batch_size: int,
    strategy: Strategy,
    hysteresis: int,
    cpu_per_process: int,
    model_name: Model,
    learning_rate: float,
    warmup: int,
    pruning_ratio_growth: float,
    pruning_max_ratio: float,
    sync_interval: int,
):
    """
    Train a model on a single rank
    """

    eps = 1e-6

    # Initial setup to handle some distribution
    torch.set_num_threads(cpu_per_process)
    if rank == 0:
        server_store = get_server_store()

    client_store = get_client_store()

    # Setup the transforms and the dataset
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)),
        ]
    )

    trainset = _DATASET(
        root="./data", train=True, download=rank == 0, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )
    print(rank, "Dataset ready")

    # Setup a model
    model = get_model(model_name, num_classes=len(trainset.targets))

    print(rank, "Model ready")
    torch.random.manual_seed(
        rank
    )  # Make sure that the processing differs in between ranks

    # Start the training
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    max_steps = epochs * len(trainset) / batch_size
    steps = 1

    def update_lr(*_):
        if steps < warmup:
            # linear warmup
            lr_mult = float(steps) / float(max(1, warmup))
            lr_mult = max(lr_mult, 1e-2)  # could be that we've not seen any yet
        else:
            # cosine learning rate decay
            progress = float(steps - warmup) / float(max(1, max_steps - warmup))
            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return lr_mult

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=update_lr,
    )

    keep_pruning = True
    model_snapshot = None
    pruning_ratio = 0.0

    for epoch in range(epochs):
        running_loss = 0.0

        for data in trainloader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # Lottery-related handling of gradients
            if strategy == Strategy.PRUNE_SORT or strategy == Strategy.PRUNE_THRESHOLD:
                freeze_pruned_weights(model, epsilon=1e-4)
            elif strategy == Strategy.PRUNE_GROW:
                # TODO: Rigging the lottery
                pass

            # Now we can go through a normal optimizer step
            optimizer.step()
            scheduler.step()
            steps += 1

            # print statistics
            running_loss += loss.item()

            if steps % 20 == 0:
                print(
                    rank,
                    "[%d, %5d] loss: %.3f | LR: %.3f"
                    % (
                        epoch,
                        steps,
                        running_loss / 2000,
                        scheduler.get_last_lr()[0],
                    ),
                )
                running_loss = 0.0

            if steps > warmup:
                if steps % sync_interval == 0 and model_snapshot is not None:
                    if strategy == Strategy.SPREAD:
                        spread_lottery_tickets(rank, world_size, model, test_model)

                    elif strategy == Strategy.PRUNE_THRESHOLD and keep_pruning:
                        pruning_ratio = sync_exchange_lottery_tickets(
                            rank,
                            model,
                            epsilon=eps,
                            max_pruning_per_layer=0.5,
                            vote_threshold=2,
                        )

                        # Rewind the non-frozen weights to the snapshot
                        rewind_model(
                            model=model, model_snapshot=model_snapshot, epsilon=eps
                        )

                    elif strategy == Strategy.PRUNE_SORT and keep_pruning:
                        pruning_ratio = pruning_ratio_growth + pruning_ratio

                        sync_exchange_lottery_tickets_sorted(
                            rank,
                            model,
                            desired_pruning_ratio=pruning_ratio,
                        )

                        # Rewind the non-frozen weights to the snapshot
                        rewind_model(
                            model=model, model_snapshot=model_snapshot, epsilon=eps
                        )

                if pruning_ratio > pruning_max_ratio:
                    keep_pruning = False
                    print(rank, "No more pruning")

                if steps % hysteresis == 0:
                    with torch.no_grad():
                        print(rank, "Saving model snapshot")
                        model_snapshot = copy.deepcopy(model)

        print(rank, " Finished Training")

    # Test the model now
    test_model(rank, model, batch_size, transform)
