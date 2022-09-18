import numpy as np
from torch.utils.data import Dataset
import os
import glob
from utils.SE3 import *
from tqdm import tqdm
import open3d as o3d



class Scan2CAD(Dataset):
    def __init__(self, num_points = 1000, partition='train' ):
        self.train_num = 1528
        self.val_num = 218
        self.test_num = 438
        self.num_points = num_points
        self.partition = partition
        self.inlier_threshold = 0.1 ** 2

        if self.partition == 'train':
            self.num = self.train_num
            self.start = 0
            self.augment = True
            self.augment_axis = 3
            self.augment_rotation = 1
            self.augment_translation = 1

        elif self.partition == 'val':
            self.num = self.val_num
            self.start = self.train_num
            self.augment = False

        elif self.partition == 'test':
            self.num = self.test_num
            self.start = self.train_num + self.val_num
            self.augment = False
        else:
            print('gg')
        self.corr = np.load('./make_dataset/correspondence_topk_score_5cm.npy')
        self.max_instance = 35
        self.dataroot = '/media/ymz/软件/PointCLM/scan2cad_hard/'


    def __getitem__(self, item):

        data = np.load(self.dataroot + 'data{:05d}.npz'.format( item + self.start ))
        trans = data['trans']
        scan = data['scan']
        cad = data['shape']
        # corr = self.corr['corr{:05d}'.format( item + self.start )]
        corr = self.corr[item + self.start]
        src_keypts = cad[corr[:,0]]
        tgt_keypts = scan[corr[:,1]]
        num_instance = trans.shape[0]

        src_keypts += np.random.rand(src_keypts.shape[0], 3) * 0.005
        tgt_keypts += np.random.rand(tgt_keypts.shape[0], 3) * 0.005

        if self.augment:
            aug_R = rotation_matrix(self.augment_axis, self.augment_rotation)
            aug_T = translation_matrix(self.augment_translation)
            src_keypts, final_trans = self.integrate_trans_with_aug(src_keypts, trans, aug_R, aug_T)
        else:
            final_trans = trans

        corr = np.concatenate(( src_keypts , tgt_keypts ), axis = -1)

        if self.partition != 'train':
            np.random.seed(item)

        if corr.shape[0] >= self.num_points:
            np.random.shuffle(corr)
            corr = corr[:self.num_points,:]
        else:
            corr_pad = corr.repeat(self.num_points//corr.shape[0],0)
            np.random.shuffle(corr_pad)
            corr = np.concatenate((corr,corr_pad),axis = 0)
            corr = corr[:self.num_points, :]

        label = self.make_label(corr, final_trans)
        final_trans, label = self.padding_trans_label(final_trans, label)

        return (corr - corr.mean(0)).astype(np.float32), corr[:, :3].astype(np.float32), corr[:, 3:].astype(
            np.float32), label.astype(bool), final_trans.astype(np.float32)


    def __len__(self):
        return self.num

    def integrate_trans_with_aug(self, src, trans, aug_R, aug_T):


        T_inv = np.eye(4)
        T_inv[:3, 3:4] = -aug_T
        R_inv = np.eye(4)
        R_inv[:3, :3] = aug_R.T

        src_aug = (np.matmul(aug_R, src.T) + aug_T).T

        aug_inv = np.matmul(R_inv, T_inv)
        final_trans = np.matmul(trans, aug_inv.reshape(1, 4, 4).repeat(trans.shape[0], 0))


        return src_aug, final_trans

    def make_label(self, corr, final_trans):
        src = corr[:,:3].reshape(1,-1,3).repeat(final_trans.shape[0], 0)
        tgt = corr[:,3:].reshape(1,-1,3).repeat(final_trans.shape[0], 0)

        src_warped = (np.matmul(final_trans[:,:3,:3], src.transpose(0, 2, 1)) + final_trans[:,:3,3:4]).transpose(0, 2, 1)
        label = (((src_warped - tgt)**2).sum(-1) < self.inlier_threshold)

        return label

    def padding_trans_label(self, final_trans, label):
        num_instance = final_trans.shape[0]
        if num_instance == self.max_instance:
            return final_trans, label
        else:
            trans_padding = np.zeros([self.max_instance - num_instance, 4, 4])
            label_padding = np.zeros([self.max_instance - num_instance, self.num_points])
            final_trans = np.concatenate((final_trans, trans_padding),axis = 0)
            label = np.concatenate((label, label_padding),axis = 0)

            return final_trans, label





if __name__ == '__main__':
    dataset = Scan2CAD(partition='train')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, drop_last=True)
    for corr, src_kpts, tgt_kpts, label, trans in tqdm(dataloader):
        break

    label = label.squeeze()
    point_tgt = np.zeros([1, 3])
    point_outlier = np.zeros([1, 3])
    for i in range((label.sum(-1) > 1).sum()):
        point_tgt = np.concatenate((tgt_kpts.squeeze().numpy()[label[i]], point_tgt), axis=0)
        point_outlier = np.concatenate((tgt_kpts.squeeze().numpy()[~label[i]], point_outlier), axis=0)
    point_tgt = np.array(point_tgt).reshape(-1, 3)
    point_outlier = np.array(point_outlier).reshape(-1, 3)

    # src
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(src_kpts.squeeze().numpy())
    pcd1.paint_uniform_color([1, 0, 0])

    # tgt
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(point_tgt + 0.01)
    pcd2.paint_uniform_color([0, 0, 1])

    # outlier
    pcd3 = o3d.geometry.PointCloud()
    pcd3.points = o3d.utility.Vector3dVector(point_outlier)
    pcd3.paint_uniform_color([0, 1, 0])

    print((label.sum(-1) > 1).sum())
    o3d.visualization.draw_geometries([pcd1, pcd2, pcd3])