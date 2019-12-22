from __future__ import absolute_import


import torch
import math
import torch.nn.functional as F
from torch import nn
from scipy.special import binom
from torch.autograd import Variable
from torch import optim
import time
class SVFC1(torch.nn.Module):
    def __init__(self, feat_dim, num_class, is_am=True, margin=0.45, mask=1.12, scale=32):
        super(SVFC1, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(feat_dim, num_class))
        # initial kernel
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.margin = margin
        self.mask = mask
        self.scale = scale
        self.is_am = is_am
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.mm = self.sin_m * margin
        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, label):  # x (M, K), w(K, N), y = xw (M, N), note both x and w are already l2 normalized.
        kernel_norm = F.normalize(self.weight, dim=0)
        cos_theta = torch.mm(x, kernel_norm)

        batch_size = label.size(0)

        gt = cos_theta[torch.arange(0, batch_size), label].view(-1, 1)  # get ground truth score of cos distance
        if self.is_am:  # AM
            mask = cos_theta > gt - self.margin
            final_gt = torch.where(gt > self.margin, gt - self.margin, gt)

        else:  # arcface
            sin_theta = torch.sqrt(1.0 - torch.pow(gt, 2).clamp(min=1e-12))
            cos_theta_m = gt * self.cos_m - sin_theta * self.sin_m  # cos(gt + margin)
            mask = cos_theta > cos_theta_m
            final_gt = torch.where(gt > 0.0, cos_theta_m, gt)
        # process hard example.
        # mask.numpy()
        # hard_example = cos_theta[mask]
        print(mask)

        cos_theta[mask] = self.mask *cos_theta[mask] + self.mask - 1.0
        cos_theta.scatter_(1, label.data.view(-1, 1), final_gt)
        cos_theta =cos_theta*self.scale

        return cos_theta





class SVFC(nn.Module):
    """
        Refer to paper:
        Hao Wang, Yitong Wang, Zheng Zhou, Xing Ji, Dihong Gong, Jingchao Zhou,Zhifeng Li, and Wei Liu
        CosFace: Large Margin Cosine Loss for Deep Face Recognition. CVPR2018
        re-implement by yirong mao
        2018 07/02
        """

    def __init__(self, feat_dim, num_class, is_am=True, margin=0.2, mask=0.1,scale=32.00, use_gpu=True ):
        super(SVFC, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_class
        self.m = margin
        self.s = scale
        self.centers = nn.Parameter(torch.Tensor(num_class, feat_dim))
        self.ce = nn.CrossEntropyLoss()
        self.centers.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.epsilon = mask
        self.logsoftmax = nn.LogSoftmax(dim=1)
    def forward(self, feat, label):
        batch_size = feat.shape[0]
        norms = torch.norm(feat, p=2, dim=-1, keepdim=True)
        nfeat = torch.div(feat, norms)

        norms_c = torch.norm(self.centers, p=2, dim=-1, keepdim=True)
        ncenters = torch.div(self.centers, norms_c)
        logits = torch.matmul(nfeat, torch.transpose(ncenters, 0, 1))

        y_onehot = torch.zeros(logits.size()).scatter_(1,label.long().unsqueeze(1).data.cpu(),self.m)
        y_onehot = y_onehot.cuda()
        margin_logits = self.s * (logits - y_onehot)
        return margin_logits


if __name__ == '__main__':
    use_gpu = True
    fc_net = SVFC(feat_dim=512, num_class=16, is_am=True, margin=0.45, mask=1.12, scale=32)
    # features = torch.randn(4, 512)
    # targets = torch.Tensor([0, 1, 2, 3])

    #targets = torch.Tensor([0, 1, 2, 3, 2, 3, 1, 4, 5, 3, 2, 1, 0, 0, 5, 4]).long()
    fc_net = torch.nn.DataParallel(fc_net).cuda()
    if use_gpu:
        features = torch.rand(16, 512).cuda()
        targets = torch.Tensor([0, 1, 2, 3, 2, 3, 1, 4, 5, 3, 2, 1, 0, 0, 5, 4]).cuda()

    parameters = [p for p in fc_net.parameters() if p.requires_grad]
    optimizer = optim.SGD(parameters, lr=0.1, momentum=0.9, weight_decay=1e-4)

    with torch.autograd.set_detect_anomaly(True):
        fc = fc_net.forward(features, targets.long())
        log_probs = nn.LogSoftmax(dim=1)(fc)
        targets = torch.zeros(log_probs.size()).scatter_(1,targets.long().unsqueeze(1).data.cpu(),1)
        if use_gpu: targets=targets.cuda()
        loss = (-targets *log_probs).mean(0).sum()


        # loss = cluster_loss(features, targets.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(loss)
