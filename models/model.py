import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.common import knn, rigid_transform_3d
from utils.SE3 import transform
from data.modelnet40 import Synthetic_corr
from tqdm import tqdm


class NonLocalBlock(nn.Module):
    def __init__(self, num_channels=128, num_heads=1):
        super(NonLocalBlock, self).__init__()
        self.fc_message = nn.Sequential(
            nn.Conv1d(num_channels, num_channels//2, kernel_size=1),
            nn.BatchNorm1d(num_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_channels//2, num_channels//2, kernel_size=1),
            nn.BatchNorm1d(num_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_channels//2, num_channels, kernel_size=1),
        )
        self.projection_q = nn.Conv1d(num_channels, num_channels, kernel_size=1)
        self.projection_k = nn.Conv1d(num_channels, num_channels, kernel_size=1)
        self.projection_v = nn.Conv1d(num_channels, num_channels, kernel_size=1)
        self.num_channels = num_channels
        self.head = num_heads

    def forward(self, feat, attention):
        """
        Input:
            - feat:     [bs, num_channels, num_corr]  input feature
            - attention [bs, num_corr, num_corr]      spatial consistency matrix
        Output:
            - res:      [bs, num_channels, num_corr]  updated feature
        """
        bs, num_corr = feat.shape[0], feat.shape[-1]
        Q = self.projection_q(feat).view([bs, self.head, self.num_channels // self.head, num_corr])
        K = self.projection_k(feat).view([bs, self.head, self.num_channels // self.head, num_corr])
        V = self.projection_v(feat).view([bs, self.head, self.num_channels // self.head, num_corr])
        feat_attention = torch.einsum('bhco, bhci->bhoi', Q, K) / (self.num_channels // self.head) ** 0.5
        # combine the feature similarity with spatial consistency
        weight = torch.softmax(attention[:, None, :, :] * feat_attention, dim=-1)
        message = torch.einsum('bhoi, bhci-> bhco', weight, V).reshape([bs, -1, num_corr])
        message = self.fc_message(message)
        res = feat + message
        return res

class NonLocalNet(nn.Module):
    def __init__(self, in_dim=6, num_layers=6, num_channels=128):
        super(NonLocalNet, self).__init__()
        self.num_layers = num_layers

        self.blocks = nn.ModuleDict()
        self.layer0 = nn.Conv1d(in_dim, num_channels, kernel_size=1, bias=True)
        for i in range(num_layers):
            layer = nn.Sequential(
                nn.Conv1d(num_channels, num_channels, kernel_size=1, bias=True),
                # nn.InstanceNorm1d(num_channels),
                nn.BatchNorm1d(num_channels),
                nn.ReLU(inplace=True)
            )
            self.blocks[f'PointCN_layer_{i}'] = layer
            self.blocks[f'NonLocal_layer_{i}'] = NonLocalBlock(num_channels)

    def forward(self, corr_feat, corr_compatibility):
        """
        Input:
            - corr_feat:          [bs, in_dim, num_corr]   input feature map
            - corr_compatibility: [bs, num_corr, num_corr] spatial consistency matrix
        Output:
            - feat:               [bs, num_channels, num_corr] updated feature
        """
        feat = self.layer0(corr_feat)
        for i in range(self.num_layers):
            feat = self.blocks[f'PointCN_layer_{i}'](feat)
            feat = self.blocks[f'NonLocal_layer_{i}'](feat, corr_compatibility)
        return feat

class PointMI(nn.Module):
    def __init__(self,
                 in_dim=6,
                 num_layers=12,
                 num_channels=128,
                 num_iterations=10,
                 ratio=0.1,
                 inlier_threshold=0.05,
                 sigma_d=0.05,
                 k=40,
                 nms_radius=0.10,
                 ):
        super(PointMI, self).__init__()
        self.num_iterations = num_iterations # maximum iteration of power iteration algorithm
        self.ratio = ratio # the maximum ratio of seeds.
        self.num_channels = num_channels
        self.inlier_threshold = inlier_threshold
        self.sigma = nn.Parameter(torch.Tensor([1.0]).float(), requires_grad=True)
        self.sigma_spat = nn.Parameter(torch.Tensor([sigma_d]).float(), requires_grad=False)
        self.k = k # neighborhood number in NSM module.
        self.nms_radius = nms_radius # only used during testing
        self.encoder = NonLocalNet(
            in_dim=in_dim,
            num_layers=num_layers,
            num_channels=num_channels,
        )
        #
        # self.classification = nn.Sequential(
        #     nn.Conv1d(num_channels, 32, kernel_size=1, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(32, 32, kernel_size=1, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(32, 1, kernel_size=1, bias=True),
        # )

        # initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_normal_(m.weight, gain=1)
            elif isinstance(m, (nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # add gradient clipping
        # grad_clip_norm = 100
        # for p in self.parameters():
        #     p.register_hook(lambda grad: torch.clamp(grad, -grad_clip_norm, grad_clip_norm))

    def forward(self, corr_pos, src_keypts, tgt_keypts ):
        """
        Input:
            - corr_pos:   [bs, num_corr, 6]
            - src_keypts: [bs, num_corr, 3]
            - tgt_keypts: [bs, num_corr, 3]
            - testing:    flag for test phase, if False will not calculate M and post-refinement.
        Output: (dict)
            - final_trans:   [bs, 4, 4], the predicted transformation matrix.
            - final_labels:  [bs, num_corr], the predicted inlier/outlier label (0,1), for classification loss calculation.
            - M:             [bs, num_corr, num_corr], feature similarity matrix, for SM loss calculation.
            - seed_trans:    [bs, num_seeds, 4, 4],  the predicted transformation matrix associated with each seeding point, deprecated.
            - corr_features: [bs, num_corr, num_channels], the feature for each correspondence, for circle loss calculation, deprecated.
            - confidence:    [bs], confidence of returned results, for safe guard, deprecated.
        """
        # corr_pos, src_keypts, tgt_keypts = data['corr_pos'], data['src_keypts'], data['tgt_keypts']
        # bs, num_corr = corr_pos.shape[0], corr_pos.shape[1]
        # testing = 'testing' in data.keys()

        #################################
        # Step1: extract feature for each correspondence
        #################################
        with torch.no_grad():
            src_dist = torch.norm((src_keypts[:, :, None, :] - src_keypts[:, None, :, :]), dim=-1)
            corr_compatibility = src_dist - torch.norm((tgt_keypts[:, :, None, :] - tgt_keypts[:, None, :, :]), dim=-1)
            corr_compatibility = torch.clamp(1.0 - corr_compatibility ** 2 / self.sigma_spat ** 2, min=0)

        corr_features = self.encoder(corr_pos.permute(0,2,1), corr_compatibility).permute(0, 2, 1)
        normed_corr_features = F.normalize(corr_features, p=2, dim=-1)

        return  normed_corr_features, corr_compatibility



if __name__ == '__main__':
    dataset = Synthetic_corr()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
    # train = ModelNet40(1024)
    # test = ModelNet40(1024, 'test')
    model = PointMI().cuda()
    for corr, src_kpts, tgt_kpts, label, trans in tqdm(dataloader):
        corr = corr.cuda()
        src_kpts = src_kpts.cuda()
        tgt_kpts = tgt_kpts.cuda()
        label = label.cuda()
        normed_corr_features, corr_compatibility = model(corr, src_kpts, tgt_kpts)
        break


