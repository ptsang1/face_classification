import torch
from sklearn.datasets import fetch_olivetti_faces
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


def load_fetch_olivetti_faces():
    """
    :argument: no args
    :return: A dict includes images and corresponding labels
    """
    dataset = fetch_olivetti_faces()
    return {'images': dataset.images.reshape(400, -1),
            'labels': dataset.target}


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

    return train_loader, test_loader


def load_images_from_folder(folder, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))]), batch_size=1, shuffle=True, num_workers=0, **kwangs):
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
    return DataLoader(dataset, batch_size, shuffle, num_workers=num_workers)


def load_CelebA_data(batch_size=32, test_batch_size=32, transform=transforms.Compose([
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


# class FaceDataset(Dataset):
#     def __init__(self, xs, ys, label_map, n_classes, transform):
#         self.xs = xs
#         self.ys = ys
#         self.label_map = label_map
#         self.transform = transform
#         self._n_classes = n_classes
#         y = 0
#
#     def __len__(self):
#         return len(self.ys)
#
#     def __get_n_classes__(self):
#         return self._n_classes
#
#     def __getitem__(self, idx):
#         return self.transform(self.xs[idx]), torch.tensor([self.ys[idx]], dtype=torch.long)
