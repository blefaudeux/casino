import torchvision
from enum import Enum, auto
import torch


class Model(Enum):
    Resnet18 = auto()
    VGG11 = auto()


def get_model(model_name: Model, num_classes: int):
    torch.random.manual_seed(
        42
    )  # make sure that all the ranks have the same weights to begin with

    model = {
        Model.Resnet18: torchvision.models.resnet18,
        Model.VGG11: torchvision.models.vgg13,
    }[model_name](pretrained=False, progress=False, num_classes=num_classes)

    return model
