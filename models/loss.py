import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.common import knn, rigid_transform_3d
from utils.SE3 import transform
from data.modelnet40 import Synthetic_corr
from tqdm import tqdm
from models.model import PointMI
import time

def pdist(A, B, dist_type='L2'):
  if dist_type == 'L2':
    D2 = torch.sum((A.unsqueeze(1) - B.unsqueeze(0)).pow(2), 2)
    return torch.sqrt(D2 + 1e-7)
  elif dist_type == 'SquareL2':
    return torch.sum((A.unsqueeze(1) - B.unsqueeze(0)).pow(2), 2)
  else:
    raise NotImplementedError('Not implemented')


class MICLoss(nn.Module):
    def __init__(self,
                 partten = None
                 ):
        super(MICLoss, self).__init__()
        # self.sigma = nn.Parameter(torch.Tensor([1.0]).float(), requires_grad=True)
        # self.sigma_spat = nn.Parameter(torch.Tensor([sigma_d]).float(), requires_grad=False)
        self.tem = 1.0
        self.wp = 0.5
        self.parttern = partten


    def forward(self, normed_corr_features,label):


        num_instance = label.shape[1]
        batch_size = normed_corr_features.shape[0]
        loss_p = torch.zeros(batch_size, num_instance).cuda()
        loss_n = torch.zeros(batch_size, num_instance).cuda()
        # loss_p = 0
        # loss_n = 0

        for i in range(batch_size):
            for j in range(num_instance):
                if label[i][j].sum()<5:
                    continue
                ft_p = normed_corr_features[i][label[i][j]]
                ft_n = normed_corr_features[i][~label[i][j]]

                sim_p = torch.matmul(ft_p, ft_p.permute(1, 0))
                if self.parttern == 'standard':
                    loss_p[i][j] = -sim_p.mean()
                else:
                    label_p = torch.ones_like(sim_p)
                    loss_p[i][j] = torch.nn.functional.mse_loss(sim_p, label_p)


                # loss_p += torch.nn.functional.mse_loss(sim_p, label_p)

                #random select pos and neg pairs
                #
                sim_n = torch.matmul(ft_p, ft_n.permute(1, 0))
                loss_n[i][j] = torch.log( torch.exp(sim_n/self.tem).sum(-1) ).mean()
                # loss_n += torch.log( torch.exp(sim_n/self.tem).sum(-1) ).mean()

        loss = self.wp*loss_p.mean() + (1 - self.wp)*loss_n.mean()

        return loss


class HardestTripletLoss(nn.Module):
    def __init__(self,
                 mp = 0.1,
                 mn = 1.0
                 ):
        super(HardestTripletLoss, self).__init__()

        self.mp = mp
        self.mn = mn


    def forward(self, normed_corr_features,label):


        num_instance = label.shape[1]
        batch_size = normed_corr_features.shape[0]
        loss_p = torch.zeros(batch_size, num_instance).cuda()
        loss_n = torch.zeros(batch_size, num_instance).cuda()

        for i in range(batch_size):
            for j in range(num_instance):
                if label[i][j].sum()<10:
                    continue
                ft_p = normed_corr_features[i][label[i][j]]
                ft_n = normed_corr_features[i][~label[i][j]]

                pos_dist = pdist(ft_p,ft_p)
                neg_dist = pdist(ft_p,ft_n)

                neg_hardest_dist, neg_hardest_idx= neg_dist.min(-1)

                mask_pos = pos_dist > self.mp
                mask_neg = neg_hardest_dist < self.mn

                loss_p[i][j] = (F.relu(pos_dist[mask_pos] - self.mp)).pow(2).mean()
                loss_n[i][j] = (F.relu(self.mn - neg_hardest_dist[mask_neg])).pow(2).mean()


        mask_p = loss_p > 0
        mask_n = loss_n > 0


        loss = loss_p[mask_p].mean() + loss_n[mask_n].mean()

        return loss


def evaluate(normed_corr_features,label):

    with torch.no_grad():
        num_instance = label.shape[1]
        batch_size = normed_corr_features.shape[0]
        sim_pos = torch.zeros(batch_size, num_instance).cuda()
        sim_neg = torch.zeros(batch_size, num_instance).cuda()
        count = 0

        for i in range(batch_size):
            for j in range(num_instance):
                if label[i][j].sum()<5:
                    continue
                ft_p = normed_corr_features[i][label[i][j]]
                ft_n = normed_corr_features[i][~label[i][j]]

                sim_p = torch.matmul(ft_p, ft_p.permute(1, 0))
                sim_pos[i][j] = sim_p.mean()
                # loss_p += torch.nn.functional.mse_loss(sim_p, label_p)

                #random select pos and neg pairs
                #
                sim_n = torch.matmul(ft_p, ft_n.permute(1, 0))
                sim_neg[i][j] = sim_n.mean()

                count += 1

    return sim_pos.sum()/count , sim_neg.sum()/count


def evaluate_topk_dist(normed_corr_features,label):
    with torch.no_grad():
        K = 4
        num_instance = label.shape[1]
        batch_size = normed_corr_features.shape[0]
        sim_pos = torch.zeros(batch_size, num_instance).cuda()
        sim_neg = torch.zeros(batch_size, num_instance, K).cuda()
        count = 0

        for i in range(batch_size):
            for j in range(num_instance):
                if label[i][j].sum()<10:
                    continue
                ft_p = normed_corr_features[i][label[i][j]]
                ft_n = normed_corr_features[i][~label[i][j]]
                sim_p = torch.matmul(ft_p, ft_p.permute(1, 0))
                sim_n = torch.matmul(ft_p, ft_n.permute(1, 0))
                sim_pos[i][j] = sim_p.mean()
                for k in range(0, K):
                    sim_neg[i][j][k] = sim_n.topk(k= 1+5*k)[0].mean()

                count += 1

    return sim_pos.sum()/count , sim_neg.sum(0).sum(0)/count





if __name__ == '__main__':
    dataset = Synthetic_corr()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
    # train = ModelNet40(1024)
    # test = ModelNet40(1024, 'test')
    model = PointMI().cuda()
    our_loss = HardestTripletLoss().cuda()
    for corr, src_kpts, tgt_kpts, label, trans in tqdm(dataloader):
        corr = corr.cuda()
        src_kpts = src_kpts.cuda()
        tgt_kpts = tgt_kpts.cuda()
        label = label.cuda()
        time0 = time.time()
        normed_corr_features, corr_compatibility = model(corr, src_kpts, tgt_kpts)
        time1 = time.time()
        loss = our_loss(normed_corr_features, label)
        time2 = time.time()
        loss.backward()
        time3 = time.time()
        print(time1 - time0, time2 - time1, time3 - time2)
        evaluate_topk_dist(normed_corr_features, label)
        break