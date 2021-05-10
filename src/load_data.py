import torch
from sklearn.datasets import fetch_olivetti_faces
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np

class MyDataset(Dataset):
  def __init__(self, data, transform=transforms.ToTensor()):
    super(MyDataset).__init__()
    self.data = data
    self.transform=transform
  def __len__(self):
        return len(self.data)
  def __getitem__(self, idx):
    if torch.is_tensor(idx):
        idx = idx.tolist()
    return self.transform(self.data[idx][0].reshape(64, 64)),\
           self.data[idx][1]


def load_fetch_olivetti_faces(**kwangs):
    """
    :argument: no args
    :return: A dict includes images and corresponding labels
    """
    dataset = fetch_olivetti_faces()
    images = dataset.images
    labels = dataset.target
    dataset_a_indexes = np.random.choice(400, 
                                     size=int(0.8*400), 
                                     replace=False)
    
    dataset_b_indexes = np.setdiff1d(np.arange(400), dataset_a_indexes)

    dataset_a = images[dataset_a_indexes].reshape((dataset_a_indexes.shape[0], -1))
    labels_a = labels[dataset_a_indexes]

    dataset_b = images[dataset_b_indexes].reshape((dataset_b_indexes.shape[0], -1))
    labels_b = labels[dataset_b_indexes]

    train_data = [(dataset_a[i], labels_a[i]) for i in range(len(dataset_a))]
    test_data = [(dataset_b[i], labels_b[i]) for i in range(len(dataset_b))]
    m, std = dataset_a.mean(), dataset_a.std()
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([m], [std])])
    return DataLoader(MyDataset(train_data, transform), batch_size=32, shuffle=True),\
           DataLoader(MyDataset(test_data, transform), batch_size=16, shuffle=True),\
           len(set(dataset.target))


def load_MNIST_data(batch_size=32, test_batch_size=32, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))]), **kwargs):
    """
    :param batch_size: batch size for train data
    :param test_batch_size: batch size for test data
    :param transform: transform for train data
    :param kwargs: 
    :return: train DataLoader, test DataLoader
    """
    train_loader = DataLoader(
        datasets.MNIST(root='../data/', download=True, transform=transform),
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        datasets.MNIST(root='../data/', download=True, train=False, transform=transform),
        batch_size=test_batch_size,
        shuffle=False
    )

    return train_loader, test_loader, 10


def load_images_from_folder(folder, transform=transforms.Compose([
        # transforms.RandomSizedCrop(128),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.CenterCrop(64),
        transforms.Normalize((0.63450485,0.4698216,0.38891008), (0.25789934,0.22389534,0.211179))
        ]), batch_size=1, test_batch_size=1, shuffle=True, num_workers=8, train=True, **kwangs):
    """
    :param folder: data folder
    :param transform:
    :param batch_size:
    :param shuffle:
    :param num_workers:
    :param kwangs:
    :return:
    """
    dataset = datasets.ImageFolder(folder, transform)
    if train:
      return DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=num_workers)
    else:
      return DataLoader(dataset, test_batch_size, num_workers=num_workers)


def load_CelebA_data(batch_size=32, test_batch_size=32, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))]), **kwargs):
    """
    :param batch_size: batch size for train data
    :param test_batch_size: batch size for test data
    :param transform: transform for train data
    :param kwargs:
    :return: train DataLoader, test DataLoader
    """
    train_loader = DataLoader(
        datasets.CelebA(root='../data/', download=True, transform=transform),
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        datasets.MNIST(root='../data/', split='test', download=True, transform=transform),
        batch_size=test_batch_size,
        shuffle=False
    )

    return train_loader, test_loader
