import os
import sys
import glob
import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
import os
import glob
from utils.SE3 import *
from tqdm import tqdm
import open3d as o3d


def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))

def load_data(partition):
    #download()
    #BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = '/media/ymz/软件/PointCLM/'
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label

class Synthetic_corr(Dataset):
    def __init__(self, num_points = 1000, partition='train', gaussian_noise=True, outlier_rate=0.95):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.num_per_pcd = 256
        self.num_instance = 10
        self.partition = partition
        self.gaussian_noise = gaussian_noise
        self.label = self.label.squeeze()
        self.outlier_rate = outlier_rate
        self.augment_axis = 3
        self.augment_rotation = 1
        self.augment_translation = 5
        self.inlier_threshold = 0.05**2


    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_per_pcd]

        # src_keypts = pointcloud

        src_keypts = pointcloud.reshape(1,self.num_per_pcd,3).repeat(self.num_instance,0)
        tgt_keypts = src_keypts + np.clip(0.01 * np.random.randn(self.num_instance, self.num_per_pcd,3), -1 * 0.05, 0.05)
        #pointcloud num_instance, num_per_pcd, 3 :    10,256,3

        aug_R, aug_T = self.produce_augment()
        aug_trans = self.integrate_trans(aug_R, aug_T)

        tgt_keypts = self.transform(tgt_keypts, aug_trans)
        num_instance_drop = int(6 * np.random.rand() // 1)
        num_instance_input = int(self.num_instance - num_instance_drop)

        tgt_keypts[num_instance_input:] = 1.2*self.augment_translation*np.random.rand(num_instance_drop,self.num_per_pcd,3)
        corr_in = np.concatenate((src_keypts, tgt_keypts),axis = -1).reshape(-1,6)

        num_outlier_instance = int(1//(1 - self.outlier_rate) - self.num_instance) + 1
        corr_out = np.concatenate((pointcloud.reshape(1,self.num_per_pcd,3).repeat(num_outlier_instance,0) , 1.2*self.augment_translation*np.random.rand(num_outlier_instance,self.num_per_pcd,3)),axis = -1).reshape(-1,6)
        # corr_out = np.concatenate((src_keypts, 1.2*self.augment_translation*np.random.rand(self.num_instance,self.num_per_pcd,3)),axis= - 1).reshape(-1,6)

        corr = np.concatenate((corr_in,corr_out), axis = 0)
        np.random.shuffle(corr)
        corr = corr[:self.num_points]
        label = self.make_label(corr[:,:3], corr[:,3:], aug_trans, num_instance_input)
        aug_trans[num_instance_input:] = 0

        return (corr-corr.mean(0)).astype(np.float32), corr[:,:3].astype(np.float32), corr[:,3:].astype(np.float32), label.astype(bool), aug_trans.astype(np.float32)

    def __len__(self):
        return self.data.shape[0]

    def produce_augment(self):
        aug_R = []
        aug_T = []
        for i in range(self.num_instance):
            aug_R.append(rotation_matrix(self.augment_axis, self.augment_rotation))
            aug_T.append(translation_matrix(self.augment_translation))
        aug_R = np.array(aug_R)
        aug_T = np.array(aug_T)
        return aug_R, aug_T

    def integrate_trans(self, R, t):
        """
        Integrate SE3 transformations from R and t, support torch.Tensor and np.ndarry.
        Input
            - R: [3, 3] or [bs, 3, 3], rotation matrix
            - t: [3, 1] or [bs, 3, 1], translation matrix
        Output
            - trans: [4, 4] or [bs, 4, 4], SE3 transformation matrix
        """
        trans = np.zeros([self.num_instance,4,4])
        trans[:, :3, :3] = R
        trans[:, :3, 3:4] = t
        return trans

    def transform(self, pts, trans):
        """
        Applies the SE3 transformations, support torch.Tensor and np.ndarry.  Equation: trans_pts = R @ pts + t
        Input
            - pts: [num_pts, 3] or [bs, num_pts, 3], pts to be transformed
            - trans: [4, 4] or [bs, 4, 4], SE3 transformation matrix
        Output
            - pts: [num_pts, 3] or [bs, num_pts, 3] transformed pts
        """
        trans_pts = trans[:, :3, :3] @ pts.transpose(0, 2, 1) + trans[:, :3, 3:4]
        return trans_pts.transpose(0, 2, 1)

    def make_label(self, src , tgt, aug_trans, num_instance_input):
        labels = []
        for i in range(num_instance_input):
            trans = aug_trans[i]
            src_warped = (trans[ :3, :3] @ src.transpose(1, 0) + trans[ :3, 3:4]).transpose(1, 0)
            labels.append(((src_warped - tgt)**2).sum(-1) < self.inlier_threshold)

        for i in range(num_instance_input,self.num_instance):
            labels.append(np.zeros(src.shape[0]))

        return np.array(labels)


class Synthetic_corr_show(Dataset):
    def __init__(self, num_points = 1000, partition='train', gaussian_noise=True, outlier_rate=0.95):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.num_per_pcd = 256
        self.num_instance = 10
        self.partition = partition
        self.gaussian_noise = gaussian_noise
        self.label = self.label.squeeze()
        self.outlier_rate = outlier_rate
        self.augment_axis = 3
        self.augment_rotation = 1
        self.augment_translation = 5
        self.inlier_threshold = 0.05**2


    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_per_pcd]

        # src_keypts = pointcloud
        if self.partition != 'train':
            np.random.seed(item)

        src_keypts = pointcloud.reshape(1,self.num_per_pcd,3).repeat(self.num_instance,0)
        tgt_keypts = src_keypts + np.clip(0.01 * np.random.randn(self.num_instance, self.num_per_pcd,3), -1 * 0.05, 0.05)
        #pointcloud num_instance, num_per_pcd, 3 :    10,256,3

        aug_R, aug_T = self.produce_augment()
        aug_trans = self.integrate_trans(aug_R, aug_T)

        tgt_keypts = self.transform(tgt_keypts, aug_trans)
        num_instance_drop = int(6 * np.random.rand() // 1)
        num_instance_input = int(self.num_instance - num_instance_drop)

        tgt_keypts[num_instance_input:] = 1.2*self.augment_translation*np.random.rand(num_instance_drop,self.num_per_pcd,3)
        corr_in = np.concatenate((src_keypts, tgt_keypts),axis = -1).reshape(-1,6)

        num_outlier_instance = int(1//(1 - self.outlier_rate) - self.num_instance) + 1
        corr_out = np.concatenate((pointcloud.reshape(1,self.num_per_pcd,3).repeat(num_outlier_instance,0) , 1.2*self.augment_translation*np.random.rand(num_outlier_instance,self.num_per_pcd,3)),axis = -1).reshape(-1,6)
        # corr_out = np.concatenate((src_keypts, 1.2*self.augment_translation*np.random.rand(self.num_instance,self.num_per_pcd,3)),axis= - 1).reshape(-1,6)

        corr = np.concatenate((corr_in,corr_out), axis = 0)
        np.random.shuffle(corr)
        # corr = corr[:self.num_points]
        label = self.make_label(corr[:,:3], corr[:,3:], aug_trans, num_instance_input)
        aug_trans[num_instance_input:] = 0

        return (corr-corr.mean(0)).astype(np.float32), self.data[item].astype(np.float32), corr[:,3:].astype(np.float32), label.astype(bool), aug_trans.astype(np.float32)

    def __len__(self):
        return self.data.shape[0]

    def produce_augment(self):
        aug_R = []
        aug_T = []
        for i in range(self.num_instance):
            aug_R.append(rotation_matrix(self.augment_axis, self.augment_rotation))
            aug_T.append(translation_matrix(self.augment_translation))
        aug_R = np.array(aug_R)
        aug_T = np.array(aug_T)
        return aug_R, aug_T

    def integrate_trans(self, R, t):
        """
        Integrate SE3 transformations from R and t, support torch.Tensor and np.ndarry.
        Input
            - R: [3, 3] or [bs, 3, 3], rotation matrix
            - t: [3, 1] or [bs, 3, 1], translation matrix
        Output
            - trans: [4, 4] or [bs, 4, 4], SE3 transformation matrix
        """
        trans = np.zeros([self.num_instance,4,4])
        trans[:, :3, :3] = R
        trans[:, :3, 3:4] = t
        return trans

    def transform(self, pts, trans):
        """
        Applies the SE3 transformations, support torch.Tensor and np.ndarry.  Equation: trans_pts = R @ pts + t
        Input
            - pts: [num_pts, 3] or [bs, num_pts, 3], pts to be transformed
            - trans: [4, 4] or [bs, 4, 4], SE3 transformation matrix
        Output
            - pts: [num_pts, 3] or [bs, num_pts, 3] transformed pts
        """
        trans_pts = trans[:, :3, :3] @ pts.transpose(0, 2, 1) + trans[:, :3, 3:4]
        return trans_pts.transpose(0, 2, 1)

    def make_label(self, src , tgt, aug_trans, num_instance_input):
        labels = []
        for i in range(num_instance_input):
            trans = aug_trans[i]
            src_warped = (trans[ :3, :3] @ src.transpose(1, 0) + trans[ :3, 3:4]).transpose(1, 0)
            labels.append(((src_warped - tgt)**2).sum(-1) < self.inlier_threshold)

        for i in range(num_instance_input,self.num_instance):
            labels.append(np.zeros(src.shape[0]))

        return np.array(labels)





if __name__ == '__main__':
    dataset = Synthetic_corr(partition='train')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    # train = ModelNet40(1024)
    # test = ModelNet40(1024, 'test')
    for corr, src_kpts, tgt_kpts, label, trans in tqdm(dataloader):
        break

    label = label.squeeze()
    point_tgt = np.zeros([1,3])
    point_outlier = np.zeros([1,3])
    for i in range((label.sum(-1) > 1).sum()):
        point_tgt = np.concatenate((tgt_kpts.squeeze().numpy()[label[i]],point_tgt),axis = 0)
        point_outlier = np.concatenate((tgt_kpts.squeeze().numpy()[~label[i]],point_outlier),axis = 0)
    point_tgt = np.array(point_tgt).reshape(-1,3)
    point_outlier = np.array(point_outlier).reshape(-1,3)


    #src
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(src_kpts.squeeze().numpy())
    pcd1.paint_uniform_color([1,0,0])

    #tgt
    pcd2 =o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(point_tgt)
    pcd2.paint_uniform_color([0, 0, 1])

    #outlier
    pcd3 = o3d.geometry.PointCloud()
    pcd3.points = o3d.utility.Vector3dVector(point_outlier)
    pcd3.paint_uniform_color([0, 1, 0])

    print((label.sum(-1) > 1).sum())
    o3d.visualization.draw_geometries([pcd1,pcd2,pcd3])


    # pcd1 = o3d.geometry.PointCloud()
    # pcd1.points = o3d.utility.Vector3dVector(corr.squeeze()[:,:3].numpy())
    # pcd1.paint_uniform_color([1,0,0])
    # pcd2 = o3d.geometry.PointCloud()
    # pcd2.points = o3d.utility.Vector3dVector(corr.squeeze()[:,3:].numpy())
    # pcd2.paint_uniform_color([0,0,1])
    #
    # lines = []
    # k = 500
    # point = []
    # for i in range(k):
    #     lines.append([2 * i, 2 * i + 1])
    #     point.append(corr.squeeze()[:,:3].numpy()[i].tolist())
    #     point.append((corr.squeeze()[:,3:].numpy()[i]).tolist())
    #     colors = [[1, 0, 0] for i in range(len(lines))]
    # line_set = o3d.geometry.LineSet(
    #     points=o3d.utility.Vector3dVector(point),
    #     lines=o3d.utility.Vector2iVector(lines),
    # )
    # o3d.visualization.draw_geometries([line_set, pcd1, pcd2])