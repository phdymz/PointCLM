import os
import shutil
import numpy as np
from tqdm import tqdm
import json
import open3d as o3d
import quaternion
from matplotlib import pyplot as plt
import time
import torch
import argparse

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
  arg = parser.add_argument_group(name)
  arg_lists.append(arg)
  return arg

def square_distance(src, dst):
    N, _ = src.shape
    M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(1, 0))
    dist += torch.sum(src ** 2, -1).view(N, 1)
    dist += torch.sum(dst ** 2, -1).view(1, M)
    return dist

def cal_inlier_rate(xyz_cad,xyz_scan,corr,trans):
    src = xyz_cad[corr[:, 0]]
    tgt = xyz_scan[corr[:, 1]]

    src_m = src.reshape(1, -1, 3).repeat(trans.shape[0], 0)
    tgt_m = tgt.reshape(1, -1, 3).repeat(trans.shape[0], 0)

    src_m_homo = np.concatenate((src_m, np.ones([src_m.shape[0], src_m.shape[1], 1])), axis=-1)
    src_m_warp = np.matmul(trans, src_m_homo.transpose(0, 2, 1)).transpose(0, 2, 1)[:, :, :3]

    inlier_rate = (((src_m_warp - tgt_m) ** 2).sum(-1) < 0.01).sum(-1)

    return inlier_rate

def show_corr(cad,scan,corr):
    pcd0 = o3d.geometry.PointCloud()
    pcd0.points = o3d.utility.Vector3dVector(cad)
    pcd0.paint_uniform_color([1, 0, 0])

    # pcd1 = o3d.geometry.PointCloud()
    # pcd1.points = o3d.utility.Vector3dVector((np.matmul(trans_final[1][:3,:3], cad_aug.T) + trans_final[1][:3,3:4]).T[:,:3] + 0.01)
    # pcd1.paint_uniform_color([0, 0, 0])

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(scan)
    pcd2.paint_uniform_color([0, 1, 0])

    lines = []
    k = len(corr)
    point = []
    for i in range(k):
        lines.append([2 * i, 2 * i + 1])
        point.append(cad[corr[i][0]].tolist())
        point.append(scan[corr[i][1]].tolist())

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(point),
        lines=o3d.utility.Vector2iVector(lines),
    )

    o3d.visualization.draw_geometries([pcd0, pcd2,  line_set])



CORRESPONDENCE = add_argument_group('Make correspondences using calculated features')
CORRESPONDENCE.add_argument('--save_root', type=str, default='/media/ymz/软件/PointCLM/scan2cad_hard/')


if __name__ == '__main__':

    args = parser.parse_args()
    output_root = args.save_root
    use_nsm = False
    downsample_num = 20000
    num_sel = 10000
    R = 0.1 ** 2

    correspondence = []
    inlier_rates = []

    for i in tqdm(range(0, 2184)):
        #     i = 1528 + 218
        savename = output_root + 'data{:05d}.npz'.format(i)
        data = np.load(savename, allow_pickle=True)
        trans = data['trans']
        scan = data['scan']
        ft_scan = data['scan_ft']
        cad = data['shape']
        ft_cad = data['shape_ft']

        scan_gpu = torch.from_numpy(scan)
        ft_cad = torch.from_numpy(ft_cad)
        ft_scan = torch.from_numpy(ft_scan)

        if not use_nsm:
            sim_ft = (ft_scan @ ft_cad.T)
            sim_ft_scan = sim_ft.max(-1)[0]
            scan_idx = sim_ft_scan.topk(k=num_sel)[1]
            cad_idx = sim_ft[scan_idx].max(-1)[1]
            corr = torch.cat((cad_idx.unsqueeze(1), scan_idx.unsqueeze(1)), dim=1)
            corr = corr.cpu().numpy()
            correspondence.append(corr)
            inlier_rates = inlier_rates + cal_inlier_rate(cad, scan, corr, trans).tolist()

        else:
            sim_ft = (ft_scan @ ft_cad.T)
            sim_ft_scan = sim_ft.max(-1)[0]
            score_sel, scan_idx = sim_ft_scan.topk(k=downsample_num)
            scan_sel = scan_gpu[scan_idx]

            dist = square_distance(scan_sel, scan_sel)
            score_relation = score_sel.unsqueeze(1) >= score_sel.unsqueeze(0)
            score_relation = score_relation.bool() | (dist >= R).bool()
            is_local_max = score_relation.min(-1)[0]

            if is_local_max.sum() >= num_sel:
                _, idx_sel = (score_sel * is_local_max).topk(k=num_sel)
                scan_idx = scan_idx[idx_sel]
                cad_idx = sim_ft[scan_idx].max(-1)[1]
                corr = torch.cat((cad_idx.unsqueeze(1), scan_idx.unsqueeze(1)), dim=1)
                corr = corr.cpu().numpy()
            else:
                scan_idx = sim_ft_scan.topk(k=num_sel)[1]
                cad_idx = sim_ft[scan_idx].max(-1)[1]
                corr = torch.cat((cad_idx.unsqueeze(1), scan_idx.unsqueeze(1)), dim=1)
                corr = corr.cpu().numpy()

            correspondence.append(corr)
            inlier_rates = inlier_rates + cal_inlier_rate(cad, scan, corr, trans).tolist()


    inlier_rates = np.array(inlier_rates) / num_sel
    correspondence = np.array(correspondence)
    slices = [i / 100 for i in range(0, 101, 2)]
    plt.hist(inlier_rates, bins=slices)
    plt.show()

    if not use_nsm:
        np.save('correspondence_topk_score_5cm.npy', correspondence)
    else:
        np.save('correspondence_topk_score_with_nsm_R10cm.npy', correspondence)