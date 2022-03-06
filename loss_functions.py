import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

    
class AAML(nn.Module):
    def __init__(self, in_features, out_features, margin=0.5, scale=64):
        super().__init__()
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        self.out_features = out_features
        self.margin = torch.Tensor([margin]).to(device)
        self.scale = torch.Tensor([scale]).to(device)
        self.cos_m = torch.cos(self.margin).to(device)
        self.sin_m = torch.sin(self.margin).to(device)
        self.mm = torch.sin(torch.Tensor([np.pi]).to(device) - margin) * margin
        self.threshold = torch.cos(torch.Tensor([np.pi]).to(device) - margin)
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, x, t):
        x_norm = F.normalize(x)
        W_norm = F.normalize(self.weight, dim=0)

        t = t.to(torch.int64)
        one_hot = torch.nn.functional.one_hot(t,num_classes=self.out_features)
        cos = torch.matmul(x_norm, W_norm) #cosine similarity

        cos_2 = torch.pow(cos, 2)
        sin = torch.pow(1 - cos_2, 0.5)

        cos_theta_m = self.scale * (self.cos_m * cos - self.sin_m * sin) * one_hot
        clip_mask = torch.where(cos > self.threshold, cos*one_hot, cos)
        clip_out = self.scale * (cos - self.mm) * one_hot
        logits = self.scale * cos * (1 - one_hot) + torch.where(clip_mask > 0., cos_theta_m, clip_out)
        return logits

class RBFClassifier(nn.Module):
    def __init__(self, feature_dim, class_num, scale, gamma, cos_sim=False):
        super(RBFClassifier, self).__init__()
        self.feature_dim = feature_dim
        self.class_num = class_num
        self.weight = nn.Parameter(torch.FloatTensor(class_num, feature_dim))
        self.bias = nn.Parameter(torch.FloatTensor(class_num))
        self.scale = scale
        self.gamma = gamma
        nn.init.xavier_uniform_(self.weight)
        self.cos_bool = cos_sim

    def forward(self, feat):
        if(self.cos_bool == False):
            diff = torch.unsqueeze(self.weight, dim=0) - \
                torch.unsqueeze(feat, dim=1)
            diff = torch.mul(diff, diff)
            metric = torch.sum(diff, dim=-1)
        else:
            feat_norm = F.normalize(feat)
            weight_norm = F.normalize(self.weight)
            metric = torch.matmul(feat_norm, weight_norm.T)
        kernal_metric = torch.exp(-1.0 * metric / self.gamma)
        logits = self.scale * kernal_metric
        return logits

class CircleLoss(nn.Module):
    def __init__(self, in_features, out_features, scale=128, margin=0.25):
        super().__init__()
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        self.out_features = out_features
        self.scale = torch.Tensor([scale]).to(device)
        self.margin = torch.Tensor([margin]).to(device)
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, t):
        x_norm = F.normalize(x)
        W_norm = F.normalize(self.weight, dim=0)

        similarities = torch.mm(x_norm, W_norm)

        alpha_p = torch.clamp(-similarities+1+self.margin, min=0, max=None)
        alpha_n = torch.clamp(similarities+self.margin, min=0, max=None)

        delta_p = 1 - self.margin
        delta_n = self.margin

        s_p = self.scale * alpha_p * (similarities - delta_p)
        s_n = self.scale * alpha_n * (similarities - delta_n)

        t = t.to(torch.int64)
        one_hot = torch.nn.functional.one_hot(t,num_classes=self.out_features)

        logits = one_hot * s_p + (1 - one_hot) * s_n
        return logits


class Model(nn.Module):
    def __init__(self, pretrain, classifier):
        super().__init__()
        self.pretrain = pretrain
        self.classifier = classifier

    def get_embs(self, x):
        return self.pretrain(x)

    def forward(self, x):
        x = self.get_embs(x)
        x = self.classifier(x)
        return x


class ModelArc(nn.Module):
    """Same as Model() but Additive Angular Margin Loss needs target vector in classification layer
    """
    def __init__(self, pretrain, classifier):
        super().__init__()
        self.pretrain = pretrain
        self.classifier = classifier

    def get_embs(self, x):
        return self.pretrain(x)

    def forward(self, x, t):
        x = self.get_embs(x)
        x = self.classifier(x, t)
        return x
