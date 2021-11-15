import torch
import torchvision
import torchvision.transforms as transforms
from enum import Enum, auto
from spread import spread_lottery_tickets
from prune import (
    exchange_lottery_tickets,
    freeze_pruned_weights,
    rewind_model,
    exchange_lottery_tickets_sorted,
)
import copy
from collections import namedtuple

_DATASET = torchvision.datasets.MNIST  # torchvision.datasets.CIFAR10

TrainSettings = namedtuple(
    "TrainSettings", ["LR", "pruning_eps", "pruning_max_ratio", "strategy"]
)


def test_model(rank: int, model: torch.nn.Module, batch_size: int) -> float:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
    )

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
            # calculate outputs by running images through the network
            outputs = model(images)

            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print(rank, f"Accuracy of the network on the 10000 test images: {accuracy}")
    return accuracy


class Strategy(Enum):
    PRUNE_SORT = auto()
    PRUNE_THRESHOLD = auto()
    SPREAD = auto()


def train_model(
    rank: int,
    world_size: int,
    epochs: int,
    batch_size: int,
    sync_interval: int,
    strategy: Strategy,
    hysteresis: int,
):
    """
    Train a model on a single rank
    """

    cpu_per_process = 2
    eps = 1e-6
    pruning_max_ratio = 0.3
    pruning_ratio_growth = 0.05

    # Initial setup to handle some distribution
    print(rank, f"Using {cpu_per_process} cpus per process")
    torch.set_num_threads(cpu_per_process)

    # Setup the transforms and the dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
    )

    trainset = _DATASET(
        root="./data", train=True, download=rank == 0, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )
    print(rank, "Dataset ready")

    # Setup a model
    torch.random.manual_seed(
        42
    )  # make sure that all the ranks have the same weights to begin with
    model = torchvision.models.resnet18(
        pretrained=False, progress=False, num_classes=len(trainset.targets)
    )

    # Adjust the model for the MNIST dataset
    model.conv1 = torch.nn.Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )
    print(rank, "Model ready")
    torch.random.manual_seed(rank)

    # Start the training
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    keep_pruning = True
    model_snapshot = None
    pruning_ratio = 0.0

    for epoch in range(epochs):
        running_loss = 0.0

        for i, data in enumerate(trainloader, 1):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            if strategy == Strategy.PRUNE_SORT or strategy == Strategy.PRUNE_THRESHOLD:
                freeze_pruned_weights(model, epsilon=1e-4)

            optimizer.step()

            # print statistics
            running_loss += loss.item()

            if i % 20 == 0:  # print every 200 mini-batches
                print(
                    rank,
                    "[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000),
                )
                running_loss = 0.0

            if i % sync_interval == 0 and model_snapshot is not None:
                if strategy == Strategy.SPREAD:
                    spread_lottery_tickets(rank, world_size, model, test_model)

                elif strategy == Strategy.PRUNE_THRESHOLD and keep_pruning:
                    pruning_ratio = exchange_lottery_tickets(
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

                    if pruning_ratio > pruning_max_ratio:
                        keep_pruning = False
                        print(rank, "No more pruning")

                elif strategy == Strategy.PRUNE_SORT and keep_pruning:
                    pruning_ratio = pruning_ratio_growth + pruning_ratio

                    exchange_lottery_tickets_sorted(
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

            if i % hysteresis == 0:
                with torch.no_grad():
                    model_snapshot = copy.deepcopy(model)

        print(rank, " Finished Training")

    # Test the model now
    test_model(rank, model, batch_size)
