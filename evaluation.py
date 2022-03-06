import os
import sys
from pathlib import Path
import argparse
from tqdm import tqdm
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import matplotlib.pyplot as plt

from sklearn import metrics

from pytorchcv.model_provider import get_model

from loss_functions import AAML, AdditiveAngularMarginLoss, CircleLoss, RBFClassifier
from train import USE_SLURM, Model, ModelArc
from data_load import get_loader, get_channels, get_num_classes


def main():
    args = argparser()
    path = args.path.replace("'", "")
    ev_ds = args.evalds.replace("'", "") # evaluation dataset

    print(path)
    print(ev_ds)
    
    values = get_values_from_path(path)
    
    print("Load Model")
    model = create_model(values)
    device = get_device()
    model.to(device)
    model.load_state_dict(torch.load(path)["model_state_dict"])
    model.eval()
    
    print("Get Data")
    trainloader, testloader_train = get_loader(bs=values["bs"], dataset=values["dataset"], USE_SLURM=USE_SLURM)
    _, testloader_eval = get_loader(bs=values["bs"], dataset=ev_ds, USE_SLURM=USE_SLURM)

    print("Calculate Embeddings")
    embs_traindata, ys_traindata = get_embs(model, trainloader)
    embs_train, ys_train = get_embs(model, testloader_train)
    embs_eval, ys_eval = get_embs(model, testloader_eval)
    embs_eval, ys_eval = embs_eval[:10000], ys_eval[:10000]
    
    # 1. calc means for all labels
    print("Calculate means")
    means = calc_means(embs_traindata, ys_traindata)

    # 2. get shortest distance between emb and mean
    print("Get shortest distances...")
    print("...of ID test data...")
    real_id_dist = np.array([])
    for emb in embs_train:
        sh_dist, _ = get_shortest_dist(emb, means)
        real_id_dist = np.append(real_id_dist, sh_dist.cpu())
    y_true = np.zeros(len(real_id_dist))
    print("Mean ID angle: ",real_id_dist.mean())

    print("...of ID train data...")
    test_id_dist = np.array([])
    for emb in embs_traindata:
        sh_dist, _ = get_shortest_dist(emb, means)
        test_id_dist = np.append(test_id_dist, sh_dist.cpu())
    plt.subplot(3,2,1)
    plt.hist(test_id_dist, 50)
    plt.title("ID-dist Train")
    plt.xlabel("Angle")

    plt.subplot(3,2,3)
    plt.hist(real_id_dist, 50)
    plt.title("ID-dist")
    plt.xlabel("Angle")

    print("...of OOD test data...")
    real_ood_dist = np.array([])
    for emb in embs_eval:
        sh_dist, _ = get_shortest_dist(emb, means)
        real_ood_dist = np.append(real_ood_dist, sh_dist.cpu())
    y_false = np.ones(len(real_ood_dist))
    print("Mean OOD angle: ",real_ood_dist.mean())

    plt.subplot(3,2,4)
    plt.hist(real_ood_dist, 50)
    plt.title("OOD-dist")
    plt.xlabel("Angle")

    dist_scores = np.append(real_id_dist, real_ood_dist)
    y = np.append(y_true, y_false)

    # 3. calc roc and auroc
    fpr, tpr, thresholds = metrics.roc_curve(y, dist_scores)
    roc_score = metrics.roc_auc_score(y, dist_scores)

    min_ber = np.min(calc_ber(tpr,fpr))
    argmin_ber = np.argmin(calc_ber(tpr,fpr))
    print(f"Minimum BER: {min_ber}")
    print(f"   Threshold: {thresholds[argmin_ber]}")
    print(f"AUROC: {roc_score}")    
    
    # 4. calc pr and aupr
    from sklearn.metrics import auc

    precision, recall, thresholds = metrics.precision_recall_curve(y, dist_scores)
    pr_score = auc(recall, precision)
    print(f"AUPR: {pr_score}")
    plt.subplot(3,2,6)
    plt.plot(recall, precision)
    plt.title("PR curve")
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.grid(visible=True)

    plt.tight_layout()
    
    figname = ev_ds+"-evds_"+Path(path).stem+".png"
    plt.savefig("./figs/"+figname)
    exit()


def argparser():
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--path', type=ascii, default="./model/softmax_30ep_3emb.pt",
                        help='path to trained model')
    parser.add_argument('--evalds', type=ascii, default="kmnist",
                        help='the dataset you want to evaluate with.')
    return parser.parse_args()

def get_values_from_path(path):
    """Gets the filename of the path and turns the string into values. 
    The filename needs the following structure.
    filename= loss_func_string + "_" + 
               model + "_" + 
               dataset + "_" + 
               str(embs) + "embs_" + 
               opti_name + "_" + 
               str(epochs) + "ep_" + 
               str(batch_size) + "bs_" + 
               str(learning_rate) + "lr_" + 
               name

    Args:
        path (string): relative or absolute path to file

    Returns:
        dict: returns model specifications from filename
    """
    filename = Path(path).stem  # gets filename from path 
    fn_split = filename.split("_")
    
    loss_func_val = fn_split[0].replace("'", "")
    model_val = fn_split[1].replace("'", "")
    dataset_val = fn_split[2].replace("'", "")
    emb_val = int(fn_split[3].split("embs")[0]) # analog zu ep_val
    opti_val = fn_split[4].replace("'", "")
    ep_val = int(fn_split[5].split("ep")[0]) # "50ep" =split=> ["50",""] =[0]=> "50" =int()=> 50
    bs_val = int(fn_split[6].split("bs")[0])
    lr_val = float(fn_split[7].split("lr")[0])
    num_chan_val = get_channels(dataset_val) # number of channels of the ds (grayscale = 1; rgb = 3)
    num_classes_val = get_num_classes(dataset_val) # number of classes of the ds
    
    values = {
        "loss": loss_func_val,
        "model": model_val,
        "dataset": dataset_val,
        "embs": emb_val,
        "opti": opti_val,
        "epochs": ep_val,
        "bs": bs_val,
        "lr": lr_val,
        "num_chan": num_chan_val,
        "num_cl": num_classes_val,  
    }   
    return values

def create_model(values):
    """create the model basing on the calculated values.

    Args:
        values (dict): values from the get_values_from_path function

    Raises:
        ValueError: if the loss function doesnt excist

    Returns:
        torch.nn.Module: model the network originally was trained with.
    """
    pretrain = get_model(
        values["model"], num_classes=values["embs"], in_channels=values["num_chan"])
    pretrain.output = nn.BatchNorm1d(512)
    
    if values["loss"] == "softmax":
        classifier = nn.Linear(values["embs"], values["num_cl"], bias=False)
        model = Model(pretrain, classifier)
    elif values["loss"] == "arcface":
        classifier = AAML(values["embs"], values["num_cl"])
        model = ModelArc(pretrain, classifier)
    elif values["loss"] == "circle":
        classifier = CircleLoss(values["embs"], values["num_cl"])
        model = ModelArc(pretrain, classifier)
    elif values["loss"] == "rbf":
        classifier = RBFClassifier(
            values["embs"], values["num_cl"], scale=3, gamma=1)
        model = Model(pretrain, classifier)
    else:
        raise ValueError("That loss function doesn't exist!")
    
    return model

def get_device():
    """get the device you can evaluate with.

    Returns:
        torch.device: "cuda" or "cpu"
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return device

def get_embs(model, dl):
    """run data through model and get embedding before the classification layer.

    Args:
        model (nn.Module): the model
        dl (torch.utils.data.DataLoader): the dataloader of shape [img, label]

    Returns:
        torch.Tensor: Tensor with normed embedding vectors and labels
    """
    embs = []
    ys = []
    device = get_device()
    for bx, by in dl:
        with torch.no_grad():
            bx, by = bx.to(device), by.to(device)
            embs.append(model.get_embs(bx))
            ys.append(by)
    embs = torch.cat(embs)
    embs = embs / embs.norm(p=2, dim=1)[:, None]
    ys = torch.cat(ys)
    return embs, ys

def calc_means(embs, ys):
    """calculate the normalized mean vectors to every label.

    Args:
        embs (torch.Tensor): all embeddings
        ys (torch.Tensor): relative label to embeddings

    Returns:
        torch.Tensor: tensor of mean vectors while idx of means = label
    """
    device = get_device()
    means = torch.Tensor([]).to(device)
    for lbl in range(torch.max(ys)+1):
        mean_vec = torch.mean(embs[ys == lbl],dim=0)
        mean_vec_norm = F.normalize(mean_vec.unsqueeze(0)) # normalize vectors
        means = torch.cat((means, mean_vec_norm))
    return means

def get_shortest_dist(emb, means):
    """calculate the angle to the nearest mean vecor

    Args:
        emb (torch.Tensor): embedding vector (dim N)
        means (torch.Tensor): mean vectors (dim (MxN))

    Returns:
        torch.Tensor: the shortest angle to the nearest mean
        int: label of the mean vector
    """
    sh_dist = 2 ** 31 # init shortest dist to max float
    sh_y = -1 # label with the shortest dist
    for idx, mean in enumerate(means):
        dist = cos_sim(emb.unsqueeze(0), mean.unsqueeze(0))
        dist = torch.acos(dist)
        dist = torch.rad2deg(dist)
        if dist < sh_dist:
            sh_dist = dist
            sh_y = idx
    assert sh_dist>0
    return sh_dist, sh_y
        
def calc_ber(tpr, fpr):
    """calculate the balanced error rate

    Args:
        tpr (array or int): true positives
        fpr (array or int): false positives

    Returns:
        array or int: balanced error rate
    """
    return 0.5 * ((1- tpr)+fpr)

def cos_sim(vec1, vec2):
    """Calculate the cosine similarity between to vectors

    Args:
        vec1 (torch.Tensor): vector one
        vec2 (torch.Tensor): vector two

    Returns:
        torch.Tensor: cosine similarity between vec1[i] and vec2[i] for all i's
    """
    cos = nn.CosineSimilarity()
    return cos(vec1, vec2)

if __name__ == "__main__":
    main()

