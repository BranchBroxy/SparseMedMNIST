import torch

import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data

def load_medmnist(batch_size=10, data_flag = 'pathmnist'):
    import medmnist
    from medmnist import INFO, Evaluator
    print(medmnist.__version__)

    download = True
    info = INFO[data_flag]
    task = info['task']
    DataClass = getattr(medmnist, info['python_class'])
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    # load the data
    train_dataset = DataClass(split='train', transform=data_transform, download=download)
    test_dataset = DataClass(split='test', transform=data_transform, download=download)

    pil_dataset = DataClass(split='train', download=download)

    # encapsulate data into dataloader form
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=2 * batch_size, shuffle=False)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=2 * batch_size, shuffle=False, drop_last=True)

    return train_loader, test_loader


def load_mnist(batch_size):
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root='./data',
                                              train=False,
                                              transform=transforms.ToTensor())

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    return train_loader, test_loader


import torch
import numpy as np


def add_noise_to_mnist_dataset(dataset, noise_level):
    noisy_dataset = []
    for data in dataset:
        image, label = data
        # Add noise to the image
        image = image + noise_level * torch.randn(image.size())
        # Clip the image to be between 0 and 1
        image = torch.clamp(image, 0, 1)
        # Add the noisy data to the new dataset
        noisy_dataset.append((image, label))
    return noisy_dataset


