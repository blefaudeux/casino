import torch
import torchvision
import torchvision.transforms as transforms
from enum import Enum, auto
import os

_DATASET = torchvision.datasets.MNIST  # torchvision.datasets.CIFAR10


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


def spread_lottery_tickets(rank: int, size: int, model: torch.nn.Module):
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


def freeze_pruned_weights(model: torch.nn.Module, epsilon: float):
    """
    Nuke the gradients for all the weights which are small enough.
    Ideally we could fork the optimizer and make it pruning aware
    but this is the poor man's take
    """
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "weight" in name:
                tensor = p.data
                grad_tensor = p.grad
                grad_tensor = torch.where(
                    tensor.abs() < epsilon, torch.zeros_like(grad_tensor), grad_tensor
                )
                p.grad.data = grad_tensor


def exchange_lottery_tickets(
    rank: int,
    model: torch.nn.Module,
    epsilon: float,
    max_pruning_per_layer: float,
    vote_threshold: int,
):
    """
    Each agent prunes its weights, and exchanges the pruned coordinates with the others
    """

    with torch.no_grad():
        overall_pruned = 0
        overall_parameters = 0

        for name, p in model.named_parameters():
            if "weight" in name:
                # Find the local weights which should be pruned
                local_prune = p.data < epsilon

                # Share that with everyone. all_reduce requires ints
                shared_prune = local_prune.to(torch.int32)

                torch.distributed.all_reduce(
                    shared_prune, op=torch.distributed.ReduceOp.SUM
                )

                # Only keep the pruning which is suggested by enough agents
                shared_prune = shared_prune > vote_threshold

                print(
                    rank,
                    f"{torch.sum(local_prune)} pruned locally, {torch.sum(shared_prune)} pruned collectively",
                )

                # Prune following the collective knowledge
                if torch.sum(shared_prune) / p.numel() < max_pruning_per_layer:
                    p.data = torch.where(shared_prune, p.data, torch.zeros_like(p.data))

                    # Bookkeeping:
                    overall_pruned += torch.sum(shared_prune)
                    overall_parameters += p.numel()

    pruning_ratio = overall_pruned / overall_parameters

    if rank == 0:
        print(f"Model is now {pruning_ratio:.2f} pruned")

    return pruning_ratio


class Strategy(Enum):
    PRUNE = auto()
    SPREAD = auto()


def train_model(
    rank: int,
    world_size: int,
    epochs: int,
    batch_size: int,
    sync_interval: int,
    strategy: Strategy,
):
    """
    Train a model on a single rank
    """
    # TODO: parameters to pick the model, optimizer, etc..

    # Initial setup to handle some distribution
    cpu_per_process = min(2, os.cpu_count() // world_size)
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

            if strategy == Strategy.PRUNE:
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

            if i % sync_interval == 0:
                if strategy == Strategy.SPREAD:
                    spread_lottery_tickets(rank, world_size, model)

                elif strategy == Strategy.PRUNE and keep_pruning:
                    pruning_ratio = exchange_lottery_tickets(
                        rank, model, epsilon=1e-4, max_pruning_per_layer=0.5, vote_threshold=2
                    )
                    # FIXME: We should reset the non-pruned weights here I believe

                    if pruning_ratio > 0.3:
                        keep_pruning = False
                        print(rank, "No more pruning")

        print(rank, " Finished Training")

    # Test the model now
    test_model(rank, model, batch_size)
