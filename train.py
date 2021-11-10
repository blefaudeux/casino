import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from typing import Any
import os


def test_model(
    rank: int, model: torch.nn.Module, transform: Any, batch_size: int
) -> float:
    testset = torchvision.datasets.CIFAR10(
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


def train_model(rank: int, size: int, epochs: int, batch_size: int):
    """
    Train a model on a single rank
    """
    # TODO: parameters to pick the model, optimizer, etc..

    # Initial setup to handle some distribution
    torch.random.manual_seed(rank)
    cpu_per_process = max(os.cpu_count() // size, 1)
    print(rank, f"Using {cpu_per_process} cpus per process")
    torch.set_num_threads(cpu_per_process)

    # Setup the transforms and the dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=rank == 0, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    print(f"{rank} - Dataset ready")

    # Setup a model
    model = torchvision.models.resnet18(pretrained=False, progress=False)
    print(f"{rank} - Model ready")

    # Start the training
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
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
            if i % 200 == 199:  # print every 200 mini-batches
                print(
                    rank,
                    "[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000),
                )
                running_loss = 0.0

        print(rank, " Finished Training")

    # Test the model now
    test_model(rank, model, transform, batch_size)
