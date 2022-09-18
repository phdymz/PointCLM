import torch
import numpy as np
from models.model import PointMI
from data.modelnet40 import Synthetic_corr
from models.loss import MICLoss, evaluate, HardestTripletLoss, evaluate_topk_dist
from tqdm import tqdm
import open3d as o3d
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import time
from data.scan2cad import Scan2CAD
import os
import random

import argparse

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
  arg = parser.add_argument_group(name)
  arg_lists.append(arg)
  return arg


def seed_everything(seed=1000):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_epoch(net, train_loader, loss, opt):
    net.train()
    num_examples = 0
    loss_total = 0

    for corr, src_kpts, tgt_kpts, label, _  in tqdm(train_loader):

        corr = corr.cuda()
        src_kpts = src_kpts.cuda()
        tgt_kpts = tgt_kpts.cuda()
        label = label.cuda()
        batch_size = corr.shape[0]

        opt.zero_grad()
        normed_corr_features, _ = net(corr, src_kpts, tgt_kpts)
        train_loss = loss(normed_corr_features,label)

        train_loss.backward()

        do_step = True
        for param in net.parameters():
            if param.grad is not None:
                if (1 - torch.isfinite(param.grad).long()).sum() > 0:
                    do_step = False
                    break
        if do_step is True:
            opt.step()
            num_examples += batch_size
            loss_total += train_loss.item() * batch_size

    return loss_total/num_examples


def test_one_epoch(net, test_loader, loss):
    net.eval()
    num_examples = 0
    pos_dis_total = 0
    neg_dis_total = 0
    total_test_loss = 0

    for corr, src_kpts, tgt_kpts, label, _  in tqdm(test_loader):

        corr = corr.cuda()
        src_kpts = src_kpts.cuda()
        tgt_kpts = tgt_kpts.cuda()
        label = label.cuda()
        batch_size = corr.shape[0]

        normed_corr_features, _ = net(corr, src_kpts, tgt_kpts)
        # test_loss = loss(normed_corr_features, label)
        pos_dis ,neg_dis  = evaluate_topk_dist(normed_corr_features, label)

        pos_dis_total += pos_dis * batch_size
        neg_dis_total += neg_dis * batch_size
        # total_test_loss += test_loss.item() * batch_size
        num_examples += batch_size


    return pos_dis_total/num_examples , neg_dis_total/num_examples, total_test_loss/num_examples


def train(net, train_loader, test_loader, loss, boardio):

    opt = optim.SGD(net.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-6)
    # root = 'checkpoints/' + str(int(time.time()))

    for i in range(150):
        train_loss = train_one_epoch(net, train_loader, loss, opt)
        dis_mean_pos, dis_mean_neg, test_loss = test_one_epoch(net, test_loader, loss)

        print('epoch{}:'.format(i),'train_loss:',train_loss)
        print('epoch{}:'.format(i),'test_loss:',test_loss)
        print('epoch{}:'.format(i),'mean distance between postive pair:',dis_mean_pos)
        for k in range(dis_mean_neg.shape[-1]):
            print('epoch{}:'.format(i),'mean distance between top{} hardest negative pair:'.format(1+5*k),dis_mean_neg[k])

        boardio.add_scalar('trainloss',train_loss,i)
        boardio.add_scalar('testloss', test_loss, i)
        boardio.add_scalar('mean_distance_between_postive_pair',dis_mean_pos,i)
        for k in range(dis_mean_neg.shape[-1]):
            boardio.add_scalar('mean_distance_between_top{}_hardest_negative_pair:'.format(1+5*k),dis_mean_neg[k],i)

        # if torch.cuda.device_count() > 1:
        #     torch.save(net.module.state_dict(), root + 'epoch{}.pkl'.format(i))
        # else:
        #     torch.save(net.state_dict(), root + 'epoch{}.pkl'.format(i))
        # torch.save(net.state_dict(), '/home/ymz/桌面/MI/checkpoints/real_scan/checkpoints/' + 'epoch{}.pkl'.format(i))
        save_checkpoint(i, net, opt)

    print('finish')

def save_checkpoint(epoch, net, optimizer, scheduler=None):
    save_root = '/home/ymz/桌面/MI/checkpoints/real_scan/checkpoints/'
    state = {
        'epoch': epoch,
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    filename = save_root + 'epoch{}.pkl'.format(epoch)
    torch.save(state, filename)


TRAIN = add_argument_group('Training on scan2cad')
TRAIN.add_argument('--save_root', type=str, default='', help='Path to save tensorboard')


if __name__ == '__main__':
    args = parser.parse_args()

    seed_everything(seed=1)

    train_loader = DataLoader(
        Scan2CAD(partition='train'),
        batch_size= 16, shuffle=True, drop_last=True)
    val_loader = DataLoader(
        Scan2CAD(partition='val'),
        batch_size=16, shuffle=False, drop_last=False)

    net = PointMI(inlier_threshold=0.1, sigma_d=0.10).cuda()


    loss = HardestTripletLoss(mp=0.1, mn=1.4)
    # net = torch.nn.DataParallel(net, device_ids=[0,1])
    # loss = torch.nn.DataParallel(loss, device_ids=[0,1])

    root = args.save_root + str(int(time.time()))
    boardio = SummaryWriter(log_dir= root)

    train(net, train_loader, val_loader, loss, boardio)

    boardio.close()