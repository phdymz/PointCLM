import torch
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from models.model import PointMI
from data.scan2cad import Scan2CAD
import time
import open3d as o3d
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
from models.common import rigid_transform_3d
from utils.SE3 import *
from data.modelnet40 import Synthetic_corr
import argparse

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
  arg = parser.add_argument_group(name)
  arg_lists.append(arg)
  return arg



def cal_sim_sp(src_keypts, tgt_keypts, sigma_spat = 0.1):
    src_dist = torch.norm((src_keypts[:, :, None, :] - src_keypts[:, None, :, :]), dim=-1)
    corr_compatibility = src_dist - torch.norm((tgt_keypts[:, :, None, :] - tgt_keypts[:, None, :, :]), dim=-1)
    corr_compatibility = torch.clamp(1.0 - corr_compatibility ** 2 / sigma_spat ** 2, min=0)
    return corr_compatibility.squeeze(0)

def cal_cluster_num(affinity, binary = False):
    if binary:
        affinity = affinity > 0.5
    x = affinity + 1e-7      #
    D = torch.diag(x.sum(-1)**-0.5)
    L_norm = torch.eye(x.shape[0]).cuda() - torch.matmul(torch.matmul(D,x),D)
    u,s,v = torch.svd(L_norm)
    max_cha = 0
    num_cluster = 0
    for i in range(0,x.shape[0]-1):
        if s[-(2+i)] - s[-(1+i)] > max_cha :
            max_cha = s[-(2+i)] - s[-(1+i)]
            num_cluster = i
    return num_cluster + 1


def random_triplet(high, size):
    triplets = torch.randint(low=0, high=high, size=(int(size * 1.2), 3))
    local_dup_check = (triplets - triplets.roll(1, 1) != 0).all(dim=1)
    triplets = triplets[local_dup_check]
    return triplets


def ransac_torch(src_keypts, tgt_keypts, size=1000):
    with torch.no_grad():
        # reflect = torch.eye(3).cuda()

        src_keypts = src_keypts.squeeze()
        tgt_keypts = tgt_keypts.squeeze()

        num_corr = len(src_keypts)

        triplets = random_triplet(num_corr, size)
        idx0 = triplets[:, 0]
        idx1 = triplets[:, 1]
        idx2 = triplets[:, 2]

        src = torch.cat(
            (src_keypts[idx0].unsqueeze(1), src_keypts[idx1].unsqueeze(1), src_keypts[idx2].unsqueeze(1)), dim=1)
        tgt = torch.cat(
            (tgt_keypts[idx0].unsqueeze(1), tgt_keypts[idx1].unsqueeze(1), tgt_keypts[idx2].unsqueeze(1)), dim=1)

        trans_guess = rigid_transform_3d(src, tgt)

        src_keypts = src_keypts.unsqueeze(0)
        tgt_keypts = tgt_keypts.unsqueeze(0)

        pred_position = transform(src_keypts, trans_guess)

        L2_dis_square = torch.sum((pred_position - tgt_keypts) ** 2, -1)
        fitness = (L2_dis_square < 0.01).sum(-1)
        index = fitness.argmax()

        return trans_guess[index]


def cal_precision_and_recall(pred_trans, gt_trans, re_thre=15, te_thre=30):
    # gt_trans 35,4,4
    # pred_trans list
    if len(pred_trans) == 0:
        return torch.tensor(0.0), torch.tensor(0.0)

    num_instance = ((gt_trans ** 2).sum(-1).sum(-1) > 1e-6).sum()

    recall_trans = torch.zeros(num_instance)
    precision_pred = torch.zeros(len(pred_trans))

    for j, pred in enumerate(pred_trans):
        R, t = decompose_trans(pred)
        for i in range(num_instance):
            gt_R, gt_t = decompose_trans(gt_trans[i])

            re = torch.acos(torch.clamp((torch.trace(R.T @ gt_R) - 1) / 2.0, min=-1, max=1))
            te = torch.sqrt(torch.sum((t - gt_t) ** 2))

            re = re * 180 / np.pi
            te = te * 100

            if re < re_thre and te < te_thre:
                precision_pred[j] = 1
                recall_trans[i] = 1
    precision = precision_pred.sum() / len(precision_pred)
    recall = recall_trans.sum() / len(recall_trans)
    # print(num_instance, len(pred_trans), precision, recall)

    return precision, recall




TRAIN = add_argument_group('Training on scan2cad')
TRAIN.add_argument('--checkpoint_root', type=str, default='/home/ymz/桌面/MI/checkpoints/real_scan/mp0.1mn1.4seed1/checkpoints/epoch137.pkl')


if __name__ == '__main__':
    args = parser.parse_args()

    test_loader = DataLoader(
            Scan2CAD(partition='test'),
            batch_size=16, shuffle=False, drop_last=False)

    net = PointMI(inlier_threshold=0.1, sigma_d=0.10).cuda()
    device = torch.device('cuda')
    checkpoint_root = args.checkpoint_root


    miss = net.load_state_dict(
        torch.load(checkpoint_root, map_location=device)['state_dict'])

    net.eval()

    mean_precision = 0
    mean_recall = 0
    mean_time = 0
    count = 0
    use_ft = True

    output = []
    precision_all = []
    recall_all = []

    for corr, src_kpts, tgt_kpts, label, trans in tqdm(test_loader):

        corr = corr.cuda()
        src_kpts = src_kpts.cuda()
        tgt_kpts = tgt_kpts.cuda()
        label = label.cuda()
        trans = trans.cuda()

        batch_size = corr.shape[0]

        a = time.time()

        with torch.no_grad():
            normed_corr_features, corr_compatibility = net(corr, src_kpts, tgt_kpts)

        # pruning
        ft = normed_corr_features
        sim_sp = corr_compatibility
        sim_ft = torch.matmul(ft, ft.transpose(2, 1))
        sim_sp_ft = torch.mul(sim_sp, sim_ft)

        idx_sp_sel = ((sim_sp > 0.8).sum(-1) > 20)
        #     idx_sp_ft_sel = ((sim_sp_ft > 0.8).sum(-1) > 20)
        #     idx_sp_ft_sel = (torch.mul(sim_sp > 0.9,sim_ft>0.85).sum(-1) > 20)
        idx_sp_ft_sel = (torch.mul(sim_sp > 0.9, sim_ft > 0.85).sum(-1) > 20)

        mean_time += time.time() - a

        for j in range(label.shape[0]):

            pred_trans = []

            a = time.time()

            if use_ft:
                idx_sp = idx_sp_ft_sel[j]
            else:
                idx_sp = idx_sp_sel[j]

            src_kpts_sel = src_kpts[j][idx_sp].unsqueeze(0)
            tgt_kpts_sel = tgt_kpts[j][idx_sp].unsqueeze(0)
            ft_sel = normed_corr_features[j][idx_sp]

            sim_sp_sel = cal_sim_sp(src_kpts_sel, tgt_kpts_sel)
            sim_ft_sel = torch.matmul(ft_sel, ft_sel.transpose(1, 0))

            #         if use_ft:
            #             sim_sp_sel = torch.mul(sim_sp_sel,sim_ft_sel > 0.85)

            num_clusters = cal_cluster_num(sim_sp_sel)  # 经实验证明，直接用sim_sp和与二值化差别不大(97%情况下一样)

            affinity = sim_sp_sel.cpu().numpy()
            if len(affinity) > 3:
                clustering = SpectralClustering(n_clusters=num_clusters, assign_labels="discretize",
                                                affinity='precomputed').fit(affinity)
                cluster_label = clustering.labels_
                for no_cluster in range(num_clusters):
                    mask_sp = (cluster_label == no_cluster)
                    if mask_sp.sum() > 9:
                        src_set = src_kpts_sel[0][mask_sp]
                        tgt_set = tgt_kpts_sel[0][mask_sp]
                        pred_trans.append(ransac_torch(src_set, tgt_set, 50))
            #         pred_trans = torch.stack(pred_trans)

            mean_time += time.time() - a

            gt_trans = trans[j]
            precision, recall = cal_precision_and_recall(pred_trans, gt_trans, re_thre=15, te_thre=10)
            mean_precision += precision
            mean_recall += recall
            count += 1

            pred = []
            for item_in_pred in pred_trans:
                pred.append(item_in_pred.cpu().numpy())
            output.append(pred)
            precision_all.append(precision.cpu().numpy())
            recall_all.append(recall.cpu().numpy())

    output = np.array(output)
    precision_all = np.array(precision_all)
    recall_all = np.array(recall_all)
    print('percision', mean_precision / count)
    print('recall', mean_recall / count)
    print('F1 score',
          2 * (mean_precision / count) * (mean_recall / count) / ((mean_recall / count + mean_precision / count)))
    print('time', mean_time / count)