from collections import OrderedDict
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models


arch_types = {
    'densenet121': models.densenet121,
    'densenet169': models.densenet169,
    'densenet161': models.densenet161,
    'densenet201': models.densenet201,
    'vgg11': models.vgg11,
    'vgg13': models.vgg13,
    'vgg16': models.vgg16,
    'vgg19': models.vgg19,
}


def validate_arch(arch: str) -> bool:
    """
    Helper to determine if the supplied arch type is valid on our
    training program
    :param arch:
    :return:
    """
    if arch.lower() in arch_types:
        return True

    return False


def load_dataset(data_dir: str) -> (datasets.ImageFolder, torch.utils.data.DataLoader):
    data_groups = ['train', 'test', 'valid']
    data_dirs = {
        'train': data_dir + '/train',
        'test': data_dir + '/test',
        'valid': data_dir + '/valid'
    }

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.Resize(255),
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    }
    image_datasets = {x: datasets.ImageFolder(data_dirs[x], transform=data_transforms[x]) for x in data_groups}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True) for x in data_groups}

    return image_datasets, dataloaders


def create_model(arch_type: str, hidden_units: int, out_features: int):
    model = arch_types[arch_type]
    model = model(pretrained=True)

    for p in model.parameters():
        p.requires_grad = False

    model.classifier = get_classifier(units=hidden_units, out_features=out_features)

    return model


def get_criterion() -> nn.NLLLoss:
    return nn.NLLLoss()


def get_optimizer(model, state_dict=None) -> optim.Adam:
    """
    returns an instanitated Adam optimizer, and will load
    state if applicable
    """
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    if state_dict:
        optimizer.load_state_dict(state_dict)

    return optimizer


def get_classifier(units: int, out_features: int) -> None:
    return nn.Sequential(
        OrderedDict([
            ('fc1', nn.Linear(25088, units)),
            ('relu', nn.ReLU()),
            ('dropout2', nn.Dropout(p=0.5, inplace=False)),
            ('fc2', nn.Linear(units, out_features)),
            ('output', nn.LogSoftmax(dim=1))
        ]))


def validate_model(model, criterion, dataloader, device) -> (float, float):
    model.eval()
    model.to(device=device)

    accuracy, test_loss = 0, 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    return test_loss / len(dataloader), accuracy / len(dataloader)


def train_model(model, use_gpu: bool, epochs: int, dataloader) -> Tuple:
    criterion, optimizer = get_criterion(), get_optimizer(model=model)
    steps, running_loss = 0, 0
    print_every = 15
    device = torch.device("cuda" if use_gpu else "cpu")

    model.to(device)

    for e in range(epochs):
        print(f'{e+1}/{epochs} Epochs')
        running_loss = 0
        for inputs, labels in dataloader['train']:
            steps += 1
            # Move our inputs and labels to the default device, which in this example is the cpu
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            log_ps = model(inputs)

            loss = criterion(log_ps, labels)
            loss.backward()

            optimizer.step()

            # track a running total of of loss as we train the network
            running_loss += loss.item()

            if steps % print_every == 0:
                loss, accuracy = validate_model(model=model,
                                                criterion=criterion,
                                                dataloader=dataloader['valid'],
                                                device=device)
                print("Epoch: {}/{} ".format(e + 1, epochs),
                      "Training Loss: {:.3f} ".format(running_loss / print_every),
                      "Validation Loss: {:.3f} ".format(loss),
                      "Validation Accuracy: {:.3f}".format(accuracy))

                running_loss = 0

                # Put model back in training mode
                model.train()
    return model, optimizer


def save_checkpoint(model, path, image_datasets, epochs, optimizer, arch) -> None:
    model.class_to_idx = image_datasets['train']
    state = {
        'model': arch,
        'epoch': epochs,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'class_to_idx': model.class_to_idx
    }
    torch.save(state, f'{path}/checkpoint-{arch}.pth')
    print('Info -- Checkpoint saved to: ' + f'{path}/checkpoint-{arch}.pth')


def load_checkpoint():
    pass