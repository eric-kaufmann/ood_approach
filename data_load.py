from torch.utils.data import Dataset
import torch
import torchvision
import torchvision.transforms as transforms


class Noise(Dataset):
    def __init__(self, num_data=10000, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.num_data = num_data

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        image = torch.randn(3,224,224)
        label = 0
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
def get_loader(dataset, bs=64):
    path = "../data"
    if dataset == "mnist":
        transform = transforms.Compose([transforms.Resize(224),
                                        transforms.Grayscale(3),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])
        trainset = torchvision.datasets.MNIST(
            path, train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=bs, shuffle=True)
        testset = torchvision.datasets.MNIST(
            path, train=False, transform=transform)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=bs, shuffle=False)
        return trainloader, testloader
    elif dataset == "kmnist":
        transform = transforms.Compose([transforms.Resize(224),
                                        transforms.Grayscale(3),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])
        trainset = torchvision.datasets.KMNIST(
            path, train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=bs, shuffle=True)
        testset = torchvision.datasets.KMNIST(
            path, train=False, transform=transform)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=bs, shuffle=False)
        return trainloader, testloader
    elif dataset == "cifar10":
        transform = transforms.Compose([transforms.Resize(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean = [0.4914, 0.4822, 0.4465], std = [0.2470, 0.2435, 0.2616])])
                                        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                                        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        trainset = torchvision.datasets.CIFAR10(
            root=path, train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=bs, shuffle=True)
        testset = torchvision.datasets.CIFAR10(
            root=path, train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=bs, shuffle=False)
        return trainloader, testloader
    elif dataset == "cifar100":
        transform = transforms.Compose([transforms.Resize(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        trainset = torchvision.datasets.CIFAR100(
            root=path, train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=bs, shuffle=True)
        testset = torchvision.datasets.CIFAR100(
            root=path, train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=bs, shuffle=False)
        return trainloader, testloader
    elif dataset == "lsun":
        transform = transforms.Compose([transforms.Resize((224,224)),                              
                                        transforms.ToTensor()])
        trainset = torchvision.datasets.LSUN(root=path, classes="train", transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True)

        testset = torchvision.datasets.ImageFolder(path, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False)

        return trainloader, testloader
    elif dataset == "noise":
        trainset = Noise(num_data=60000)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True)

        testset = Noise(num_data=10000)
        testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=True)
        return trainloader, testloader
    elif dataset == "svhn":
        transform = transforms.Compose([transforms.Resize(224),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor()])
        trainset = torchvision.datasets.SVHN(
            root=path, split="train", download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=bs, shuffle=True)
        testset = torchvision.datasets.SVHN(
            root=path, split="test", download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=bs, shuffle=True)
        return trainloader, testloader    
    else:
        raise ValueError("dataset not found!")
    
def get_channels(dataset):
    return {
        "mnist": 3,
        "kmnist": 3,
        "cifar10": 3,
        "cifar100": 3,
        "lsun": 3,
        "noise": 1,
        "svhn": 3,
    }[dataset]
    
def get_num_classes(dataset):
    return {
        "mnist": 10,
        "kmnist": 10,
        "cifar10": 10,
        "cifar100": 100,
        "lsun": 10,
        "noise": 1,
        "svhn": 10,
        "imagenet": 1000,
    }[dataset]