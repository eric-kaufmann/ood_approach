import torch
from torch.utils.tensorboard import SummaryWriter
import sys
import os
sys.path.append(os.getcwd())

from loss_functions import RBFClassifier, AAML, CircleLoss, Model, ModelArc
from data_load import get_loader, get_channels, get_num_classes

import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
from pytorchcv.model_provider import get_model
import matplotlib.pyplot as plt
import timeit
import argparse

def main():
    # parsing arguments
    args = argParser()
    model = args.model.replace("'", "")
    dataset = args.dataset.replace("'", "")
    loss_func_string = args.loss.replace("'", "")
    opti_name = args.opti.replace("'", "")
    name = args.name.replace("'", "")
    batch_size= args.batch_size
    embs = args.embs
    epochs = args.epochs
    learning_rate = args.lr

    sched_name = "ca"

    # setting paths
    fullname = loss_func_string + "_" + model + "_" + dataset + "_" + str(embs) + "embs_" + opti_name + "_" + str(epochs) + "ep_" + str(batch_size) + "bs_" + str(learning_rate) + "lr_" + name

    PATH = "./model/" + fullname + ".pth"
    RUN_PATH = "./runs/" + fullname

    
    # print props
    print(f"model:          {model}")
    print(f"dataset:        {dataset}")
    print(f"loss function:  {loss_func_string}")
    print(f"path:           {PATH}")
    print(f"batch size:     {batch_size}")
    print(f"# embeddings:   {embs}")
    print(f"# epochs:       {epochs}")
    print(f"optimizer:      {opti_name}")
   

    # define device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # get dataloader
    trainloader, testloader = get_loader(dataset, bs=batch_size)
    in_channels = get_channels(dataset)
    num_classes = get_num_classes(dataset)
    
    # define model
    pretrain = get_model(model, num_classes=embs,
                         in_channels=in_channels)
    pretrain.output = nn.BatchNorm1d(512)
    if loss_func_string == "softmax":
        classifier = nn.Linear(embs, num_classes, bias=False)
        net = Model(pretrain, classifier)
    elif loss_func_string == "arcface":
        pretrain = get_pretrain()
        classifier = AAML(in_features=embs, out_features=num_classes, margin=0.5, scale=64)
        net = ModelArc(pretrain, classifier)
    elif loss_func_string == "rbf":
        classifier = RBFClassifier(embs, num_classes, scale=4, gamma=1.6, cos_sim=True)
        net = Model(pretrain, classifier)
    elif loss_func_string == "circle":
        pretrain = get_pretrain()
        classifier = CircleLoss(embs, num_classes, margin=0.7, scale=64)
        net = ModelArc(pretrain, classifier)
    else:
        raise ValueError("Loss function not found!")
    net.to(device)

    # define optimizer and scheduler
    if opti_name == "adam":
        opti = optim.Adam(net.parameters(), lr=learning_rate)
    elif opti_name == "sgd":
        opti = optim.SGD(net.parameters(), learning_rate, momentum=0.9, weight_decay=5e-4)
    else:
        raise ValueError("Optimizer not found!")

    # sched1 -> warmup
    sched1 = torch.optim.lr_scheduler.OneCycleLR(
        opti, 
        max_lr=learning_rate*10, 
        steps_per_epoch=10, 
        epochs=10,
        anneal_strategy='linear'
    )
    
    # sched2 -> real scheduler (ms = multi step; ca = cosine annealing)
    if sched_name == "ms":
        sched2 = torch.optim.lr_scheduler.MultiStepLR(
            opti, 
            [30,50],
            gamma=0.1
        )
    elif sched_name == "ca":
        sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opti, T_max=epochs-10)

    # define loss function
    loss_func = nn.CrossEntropyLoss()

    # train and test network
    save_epoch_acc = np.array([])
    save_epoch_loss = np.array([])
    tb = SummaryWriter(RUN_PATH)
    temp_acc = 0


    for epoch in range(epochs):
        start = timeit.default_timer()
        ep_acc = 0
        ep_loss = 0
        for idx, data in enumerate(trainloader):
            inp, label = data
            inp, label = inp.to(device), label.to(device)

            if (loss_func_string=="arcface") | (loss_func_string=="circle"):
                pred = net(inp, label) 
            else:
                pred = net(inp)
                
            opti.zero_grad()
            loss = loss_func(pred, label)
            loss.backward()
            opti.step()

            ep_loss += loss.item()
            ep_acc += torch.sum(torch.argmax(pred, dim=1)
                                == label).cpu()/len(label)
                            
        stop = stop = timeit.default_timer()
        print("Epoch: {} | Loss: {:.5f} | Acc: {:.4} | Time: {:.5f}".format(epoch+1 ,ep_loss/len(trainloader), ep_acc/len(trainloader), stop-start))
        save_epoch_acc = np.append(save_epoch_acc, ep_acc/len(trainloader))
        save_epoch_loss = np.append(save_epoch_loss, ep_loss/len(trainloader))
        
        tb.add_scalar("Loss/Train", ep_loss/len(trainloader), epoch)
        tb.add_scalar("Accuracy/Train", ep_acc/len(trainloader), epoch)

        # test on testdata
        with torch.no_grad():
            net.eval()
            ep_test_acc = 0
            ep_test_loss = 0
            for idx, data in enumerate(testloader):
                inp, label = data
                inp, label = inp.to(device), label.to(device)

                if (loss_func_string=="arcface") | (loss_func_string=="circle"):
                    pred = net(inp, label)
                else:
                    pred = net(inp)

                loss = loss_func(pred, label)
                ep_test_loss += loss.item()

                # Berechne die Accuracy des Batches
                ep_test_acc += torch.sum(torch.argmax(pred, dim=1)
                                    == label).cpu()/len(label)

            loss_res = ep_test_loss/len(testloader)
            acc_res = ep_test_acc/len(testloader)
            print(
                "   Results on Test-Data: loss: {:.5f} acc: {:.4}".format(loss_res, acc_res))
            tb.add_scalar("Loss/Test", loss_res, epoch)
            tb.add_scalar("Accuracy/Test", acc_res, epoch)
            
            if(acc_res>temp_acc):
                print("   New Best!")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': opti.state_dict(),
                    'loss': loss_res,
                    'acc': acc_res,
                    }, PATH)
                temp_acc = acc_res

            net.train()

        lr = get_lr(opti)
        tb.add_scalar("LearningRate", lr, epoch)   
        if epoch<10:
            sched1.step()
        else:
            sched2.step()
    tb.close()

    print(f"Best Accuracy on test data: {temp_acc}")


#########################
### SUPPORT FUNCTIONS ###
#########################

def argParser():
    parser = argparse.ArgumentParser(description='PyTorch Trainer')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size (default: 64)')
    parser.add_argument('--epochs', type=int, default=3,
                        help='number of epochs to train (default: 3)')
    parser.add_argument('--embs', type=int, default=128,
                        help='number of embeddings (default: 128)')
    parser.add_argument('--lr', type=float, default=0.000035,
                        help='learning rate (default: 0.000035)')
    parser.add_argument('--model', type=ascii, default="resnet10",
                        help='select pytorchcv model (default: "resnet10")')
    parser.add_argument('--loss', type=ascii, default="softmax",
                        help='select loss function model (default: "softmax")')
    parser.add_argument('--dataset', type=ascii, default="mnist",
                        help='select dataset (cifar10, mnist) (default: "mnist")')
    parser.add_argument('--name', type=ascii, default="",
                        help='adding a specific description (default: "")')
    parser.add_argument('--opti', type=ascii, default="adam",
                        help='optimizer (default: adam)')
    return parser.parse_args()


def get_lr(optimizer):
    """get learning rate from optimizer

    Args:
        optimizer (torch.optim.Optimizer): the optimizer

    Returns:
        float: the learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']

def plot_training_progress(epochs, loss, accuracy, name):
    """create training progress plots for accuracy and loss if tensorboard ins't available.

    Args:
        epochs (int): number of epochs
        loss (array): array of losses from error function from every epoch
        accuracy (array): array of accuracys for every epoch
        name (str): name your files
    """
    epochs = range(epochs)

    # accuracy
    plt.plot(epochs, accuracy, 'r-')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig("./pics/"+name+"_acc.png")
    plt.clf()

    # loss
    plt.plot(epochs, loss, 'b-')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.savefig("./pics/"+name+"_loss.png")
    plt.clf()

def get_pretrain():
    """get pretrain own trained pretrain

    Returns:
        torch.nn.Module: pretrained model
    """
    pretrain = get_model("resnet18", num_classes=512, in_channels=3)
    pretrain.output = nn.BatchNorm1d(512)
    classifier = nn.Linear(512,10, bias=False)
    model = Model(pretrain=pretrain, classifier=classifier)
    path="./model/softmax_resnet18_mnist_512embs_sgd_20ep_64bs_0.01lr.pt"
    model.load_state_dict(torch.load(path)["model_state_dict"])
    return model.pretrain


if __name__ == "__main__":
    main()