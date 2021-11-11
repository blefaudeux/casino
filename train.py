import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import os

_DATASET = torchvision.datasets.MNIST  #  torchvision.datasets.CIFAR10


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


def mix_lottery_tickets(rank: int, size: int, model: torch.nn.Module):
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


def train_model(
    rank: int, world_size: int, epochs: int, batch_size: int, sync_interval: int
):
    """
    Train a model on a single rank
    """
    # TODO: parameters to pick the model, optimizer, etc..

    # Initial setup to handle some distribution
    torch.random.manual_seed(rank)
    cpu_per_process = max(os.cpu_count() // (2 * world_size), 1)  # type: ignore
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
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    print(f"{rank} - Dataset ready")

    # Setup a model
    model = torchvision.models.resnet18(
        pretrained=False, progress=False, num_classes=len(trainset.targets)
    )

    # Adjust the model for the MNIST dataset
    model.conv1 = torch.nn.Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )
    print(f"{rank} - Model ready")

    # Start the training
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

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
                mix_lottery_tickets(rank, world_size, model)

        print(rank, " Finished Training")

    # Test the model now
    test_model(rank, model, batch_size)
