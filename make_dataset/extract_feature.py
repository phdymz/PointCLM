import os
import glob
import open3d as o3d
import numpy as np
import torch
import MinkowskiEngine as ME
from utils.pointcloud import make_point_cloud
from misc.fcgf import ResUNetBN2C as FCGF
from tqdm import tqdm
import argparse

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
  arg = parser.add_argument_group(name)
  arg_lists.append(arg)
  return arg


def extract_features(model,
                     xyz,
                     rgb=None,
                     normal=None,
                     voxel_size=0.05,
                     device=None,
                     skip_check=False,
                     is_eval=True):
    """
    Extracts FCGF features.
    Args:
        model (FCGF model instance): model used to inferr the features
        xyz (torch tensor): coordinates of the point clouds [N,3]
        rgb (torch tensor): colors, must be in range (0,1) [N,3]
        normal (torch tensor): normal vectors, must be in range (-1,1) [N,3]
        voxel_size (float): voxel size for the generation of the saprase tensor
        device (torch device): which device to use, cuda or cpu
        skip_check (bool): if true skip rigorous check (to speed up)
        is_eval (bool): flag for evaluation mode
    Returns:
        return_coords (torch tensor): return coordinates of the points after the voxelization [m,3] (m<=n)
        features (torch tensor): per point FCGF features [m,c]
    """

    if is_eval:
        model.eval()

    if not skip_check:
        assert xyz.shape[1] == 3

        N = xyz.shape[0]
        if rgb is not None:
            assert N == len(rgb)
            assert rgb.shape[1] == 3
            if np.any(rgb > 1):
                raise ValueError('Invalid color. Color must range from [0, 1]')

        if normal is not None:
            assert N == len(normal)
            assert normal.shape[1] == 3
            if np.any(normal > 1):
                raise ValueError('Invalid normal. Normal must range from [-1, 1]')

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feats = []
    if rgb is not None:
        # [0, 1]
        feats.append(rgb - 0.5)

    if normal is not None:
        # [-1, 1]
        feats.append(normal / 2)

    if rgb is None and normal is None:
        feats.append(np.ones((len(xyz), 1)))

    feats = np.hstack(feats)

    # Voxelize xyz and feats
    coords = np.floor(xyz / voxel_size)
    coords, inds = ME.utils.sparse_quantize(coords, return_index=True)
    # Convert to batched coords compatible with ME
    coords = ME.utils.batched_coordinates([coords])
    return_coords = xyz[inds]

    feats = feats[inds]

    feats = torch.tensor(feats, dtype=torch.float32)
    coords = torch.tensor(coords, dtype=torch.int32)

    stensor = ME.SparseTensor(feats, coordinates=coords, device=device)

    return return_coords, model(stensor).F

def show_correspondence(xyz_cad,xyz_scan,corr):
    pcd0 = o3d.geometry.PointCloud()
    pcd0.points = o3d.utility.Vector3dVector(xyz_cad)
    pcd0.paint_uniform_color([1, 0, 0])

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(xyz_scan)
    pcd2.paint_uniform_color([0, 0, 1])

    lines = []
    k = len(corr)
    point = []
    for i in range(k):
        lines.append([2 * i, 2 * i + 1])
        point.append(xyz_cad[corr[i][0]].tolist())
        point.append((xyz_scan[corr[i][1]]).tolist())

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(point),
        lines=o3d.utility.Vector2iVector(lines),
    )

    o3d.visualization.draw_geometries([pcd0, pcd2, line_set])


def cal_inlier_rate(xyz_cad,xyz_scan,corr,trans):
    src = xyz_cad[corr[:, 0]]
    tgt = xyz_scan[corr[:, 1]]

    src_m = src.reshape(1, -1, 3).repeat(trans.shape[0], 0)
    tgt_m = tgt.reshape(1, -1, 3).repeat(trans.shape[0], 0)

    src_m_homo = np.concatenate((src_m, np.ones([src_m.shape[0], src_m.shape[1], 1])), axis=-1)
    src_m_warp = np.matmul(trans, src_m_homo.transpose(0, 2, 1)).transpose(0, 2, 1)[:, :, :3]

    inlier_rate = ((((src_m_warp - tgt_m) ** 2).sum(-1) < 0.01).sum(0)).sum() / src_m_warp.shape[1]

    return inlier_rate


FEATURE = add_argument_group('Extract feature using pre-trained FCGF')
FEATURE.add_argument('--weight', type=str, default='/media/ymz/软件/PointCLM/pretrained/checkpoint49_lr=1e-3.pth')
FEATURE.add_argument('--output', type=str, default='/media/ymz/软件/PointCLM/scan2cad/')
FEATURE.add_argument('--save_root', type=str, default='/media/ymz/软件/PointCLM/scan2cad_hard/')


if __name__ == '__main__':
    args = parser.parse_args()

    model = FCGF(
        1,
        32,
        bn_momentum=0.05,
        conv1_kernel_size=7,
        normalize_feature=True
    ).cuda()

    checkpoint = torch.load(args.weight)

    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    output_root =args.output
    saveroot = args.save_root

    voxel_size = 0.025

    for i in tqdm(range(0, 2184)):

        savename = output_root + 'data{:05d}.npz'.format(i)
        data = np.load(savename)
        trans = data['trans']
        scan_and_cad = np.ascontiguousarray(data['scan'])
        cad_origin = np.ascontiguousarray(data['shape'])


        xyz_cad, ft_cad = extract_features(model, cad_origin, voxel_size=voxel_size)
        xyz_scan, ft_scan = extract_features(model, scan_and_cad ,voxel_size=voxel_size)

        save_path = saveroot + 'data{:05d}.npz'.format(i)
        np.savez(save_path, scan = xyz_scan, shape = xyz_cad, scan_ft = ft_scan.cpu().detach().numpy(), shape_ft = ft_cad.cpu().detach().numpy(), trans = trans)

    print('finish')

