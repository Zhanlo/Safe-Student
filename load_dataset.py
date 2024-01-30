import logging
import os

import math
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Sampler
from torchvision import datasets
import numpy as np
import torch
import torchvision.transforms as tv_transforms
from PIL import Image
import torchvision.transforms as transforms
from augmentations import *
from archive import autoaug_policy, autoaug_paper_cifar10, fa_reduced_cifar10


rng = np.random.RandomState(seed=1)

class SimpleDataset(Dataset):
    def __init__(self, dataset, mode='test', dataset_name = None):
        self.dataset = dataset
        self.mode = mode
        self.dataset_name = dataset_name

    def __getitem__(self, index):
        image = self.dataset['images'][index]
        label = self.dataset['labels'][index]

        if self.dataset_name == "CIFAR10":
            if self.mode == "train":
                self.set_aug(5)
                data0 = self.transform(Image.fromarray(image, 'RGB'))
                self.set_aug(0)
                data_noaug = self.transform(Image.fromarray(image, 'RGB'))
                return data0, data_noaug, label, index
            elif self.mode == "test":
                self.set_aug(0)
                data = self.transform(Image.fromarray(image, 'RGB'))
                return data, label, index

        elif self.dataset_name == "CIFAR100":
            if self.mode == "train":
                self.set_aug(10)
                data0 = self.transform(Image.fromarray(image, 'RGB'))
                self.set_aug(11)
                data_noaug = self.transform(Image.fromarray(image, 'RGB'))
                return data0, data_noaug, label, index
            elif self.mode == "test":
                self.set_aug(11)
                data = self.transform(Image.fromarray(image, 'RGB'))
                return data, label, index


    def set_aug(self, method):
        if method == 0:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #CIFAR10
            ])

        elif method == 1:
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        elif method == 2:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=1),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        elif method == 3:  # most widely used data augmentation for CIFAR dataset
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        elif method == 5:
            # AutoAugment & CutOut
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),  # always crop cifar10
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  #CIFAR10
            ])
            autoaug = transforms.Compose([])
            autoaug.transforms.insert(0, Augmentation(autoaug_paper_cifar10()))
            transform_train.transforms.insert(0, autoaug)
            transform_train.transforms.append(CutoutDefault(16))
            self.transform = transform_train

        elif method == 10:
            # augment for cifar100
            # AutoAugment & CutOut
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),  # CIFAR100
            ])
            autoaug = transforms.Compose([])
            autoaug.transforms.insert(0, Augmentation(autoaug_paper_cifar10()))
            transform_train.transforms.insert(0, autoaug)
            transform_train.transforms.append(CutoutDefault(16))
            self.transform = transform_train
        elif method == 11:
            # no augment for cifar100
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),  # CIFAR10
            ])


    def __len__(self):
        return len(self.dataset['images'])


class Augmentation(object):
    def __init__(self, policies):
        self.policies = policies

    def __call__(self, img):
        for _ in range(1):
            policy = random.choice(self.policies)
            for name, pr, level in policy:
                if random.random() > pr:
                    continue
                img = apply_augment(img, name, level)
        return img

class CutoutDefault(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        if self.length <= 0:
            return img
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)
        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

data_path = "./data"

def split_l_u(train_set, n_labels, n_unlabels, tot_class=6, ratio = 0.5):
    images = train_set["images"]
    labels = train_set["labels"]
    classes = np.unique(labels)


    n_labels_per_cls = n_labels // tot_class
    n_unlabels_per_cls = int(n_unlabels*(1.0-ratio)) // tot_class

    if(tot_class < len(classes)):
        n_unlabels_shift = (n_unlabels - (n_unlabels_per_cls * tot_class)) // (len(classes) - tot_class)

    l_images = []
    l_labels = []
    u_images = []
    u_labels = []


    for c in classes[:tot_class]:
        cls_mask = (labels == c)

        c_images = images[cls_mask]
        c_labels = labels[cls_mask]
        l_images += [c_images[:n_labels_per_cls]]
        l_labels += [c_labels[:n_labels_per_cls]]

        u_images += [c_images[n_labels_per_cls:n_labels_per_cls+n_unlabels_per_cls]]
        u_labels += [c_labels[n_labels_per_cls:n_labels_per_cls+n_unlabels_per_cls]]


    for c in classes[tot_class:]:

        cls_mask = (labels == c)
        c_images = images[cls_mask]

        c_labels = labels[cls_mask]
        u_images += [c_images[:n_unlabels_shift]]
        u_labels += [c_labels[:n_unlabels_shift]]


    l_train_set = {"images": np.concatenate(l_images, 0), "labels": np.concatenate(l_labels, 0)}
    u_train_set = {"images": np.concatenate(u_images, 0), "labels": np.concatenate(u_labels, 0)}


    indices = rng.permutation(len(l_train_set["images"]))
    l_train_set["images"] = l_train_set["images"][indices]
    l_train_set["labels"] = l_train_set["labels"][indices]

    indices = rng.permutation(len(u_train_set["images"]))
    u_train_set["images"] = u_train_set["images"][indices]
    u_train_set["labels"] = u_train_set["labels"][indices]

    return l_train_set, u_train_set

def split_test(test_set, tot_class=6):
    images = test_set["images"]
    labels = test_set['labels']

    classes = np.unique(labels)
    l_images = []
    l_labels = []
    for c in classes[:tot_class]:

        cls_mask = (labels == c)
        c_images = images[cls_mask]
        c_labels = labels[cls_mask]


        l_images += [c_images[:]]
        l_labels += [c_labels[:]]
    test_set = {"images": np.concatenate(l_images, 0), "labels":np.concatenate(l_labels,0)}


    indices = rng.permutation(len(test_set["images"]))
    test_set["images"] = test_set["images"][indices]
    test_set["labels"] = test_set["labels"][indices]
    return test_set


def load_mnist():
    splits = {}
    trans = tv_transforms.Compose([tv_transforms.ToPILImage(),tv_transforms.ToTensor(), tv_transforms.Normalize((0.5,), (1.0,))])
    for train in [True, False]:
        dataset = datasets.MNIST(data_path, train, transform=trans, download=True)
        data = {}
        data['images'] = dataset.data
        data['labels'] = np.array(dataset.targets)
        splits['train' if train else 'test'] = data
    return splits.values()



def load_cifar10():
    splits = {}
    for train in [True, False]:
        dataset = datasets.CIFAR10(data_path, train, download=True)
        data = {}
        data['images'] = dataset.data

        data['labels'] = np.array(dataset.targets)

        splits["train" if train else "test"] = data

    return splits.values()

def load_cifar100():
    splits = {}
    for train in [True, False]:
        dataset = datasets.CIFAR100(data_path, train, download=True)
        data = {}
        data['images'] = dataset.data
        data['labels'] = np.array(dataset.targets)
        splits["train" if train else "test"] = data
    return splits.values()




def gcn(images, multiplier=55, eps=1e-10):
    #global contrast normalization
    images = images.astype(np.float)
    images -= images.mean(axis=(1,2,3), keepdims=True)
    per_image_norm = np.sqrt(np.square(images).sum((1,2,3), keepdims=True))
    per_image_norm[per_image_norm < eps] = 1
    images = multiplier * images / per_image_norm
    return images

def get_zca_normalization_param(images, scale=0.1, eps=1e-10):
    n_data, height, width, channels = images.shape
    images = images.reshape(n_data, height*width*channels)
    image_cov = np.cov(images, rowvar=False)
    U, S, _ = np.linalg.svd(image_cov + scale * np.eye(image_cov.shape[0]))
    zca_decomp = np.dot(U, np.dot(np.diag(1/np.sqrt(S + eps)), U.T))
    mean = images.mean(axis=0)
    return mean, zca_decomp

def zca_normalization(images, mean, decomp):
    n_data, height, width, channels = images.shape
    images = images.reshape(n_data, -1)
    images = np.dot((images - mean), decomp)
    return images.reshape(n_data, height, width, channels)


def get_dataloaders(dataset, n_labels, n_unlabels, n_valid, tot_class, ratio):

    rng = np.random.RandomState(seed=1)
    dataset_name = None

    if dataset == "MNIST":
        train_set, test_set = load_mnist()
        transform = False
    elif dataset == "CIFAR10":
        dataset_name = "CIFAR10"
        train_set, test_set = load_cifar10()

        train_set['labels'] -= 2
        test_set['labels'] -= 2
        train_set['labels'][np.where(train_set['labels'] == -2)] = 8
        train_set['labels'][np.where(train_set['labels'] == -1)] = 9

        test_set['labels'][np.where(test_set['labels'] == -2)] = 8
        test_set['labels'][np.where(test_set['labels'] == -1)] = 9


    elif dataset == "CIFAR100":
        dataset_name = "CIFAR100"
        train_set, test_set = load_cifar100()
        transform = False



    #permute index of training set
    indices = rng.permutation(len(train_set['images']))
    train_set['images'] = train_set['images'][indices]
    train_set['labels'] = train_set['labels'][indices]

    #split training set into training and validation
    train_images = train_set['images'][n_valid:]
    train_labels = train_set['labels'][n_valid:]
    validation_images = train_set['images'][:n_valid]
    validation_labels = train_set['labels'][:n_valid]

    validation_set = {'images': validation_images, 'labels': validation_labels}
    train_set = {'images': train_images, 'labels': train_labels}


    #split training set into labeled and unlabeled data
    validation_set = split_test(validation_set, tot_class=tot_class)
    test_set = split_test(test_set, tot_class=tot_class)




    l_train_set, u_train_set = split_l_u(train_set, n_labels, n_unlabels, tot_class=tot_class, ratio=ratio)


    logging.info("Unlabeled data in distribuiton : {}, Unlabeled data out distribution : {}".format(
          np.sum(u_train_set['labels'] < tot_class), np.sum(u_train_set['labels'] >= tot_class)))

    l_train_set = SimpleDataset(l_train_set, mode='train', dataset_name=dataset_name)
    u_train_set = SimpleDataset(u_train_set, mode='train', dataset_name=dataset_name)
    validation_set = SimpleDataset(validation_set, mode='test', dataset_name=dataset_name)
    test_set = SimpleDataset(test_set, mode='test', dataset_name=dataset_name)


    logging.info("labeled data : {}, unlabeled data : {},  training data : {}".format(
        len(l_train_set), len(u_train_set), len(l_train_set) + len(u_train_set)))
    logging.info("validation data : {}, test data : {}".format(len(validation_set), len(test_set)))


    return l_train_set, u_train_set, validation_set, test_set






















