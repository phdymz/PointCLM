import torch
import numpy as np
from models.model import PointMI
from data.modelnet40 import Synthetic_corr
from models.loss import MICLoss, evaluate
from tqdm import tqdm
import open3d as o3d
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import time



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


def test_one_epoch(net, test_loader):
    net.eval()
    num_examples = 0
    pos_dis_total = 0
    neg_dis_total = 0

    for corr, src_kpts, tgt_kpts, label, _  in tqdm(test_loader):

        corr = corr.cuda()
        src_kpts = src_kpts.cuda()
        tgt_kpts = tgt_kpts.cuda()
        label = label.cuda()
        batch_size = corr.shape[0]

        normed_corr_features, _ = net(corr, src_kpts, tgt_kpts)
        pos_dis ,neg_dis  = evaluate(normed_corr_features, label)

        pos_dis_total += pos_dis * batch_size
        neg_dis_total += neg_dis * batch_size
        num_examples += batch_size

    return pos_dis_total/num_examples , neg_dis_total/num_examples


def train(net, train_loader, test_loader, loss, boardio):

    opt = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-6)
    # root = 'checkpoints/' + str(int(time.time()))

    for i in range(50):
        train_loss = train_one_epoch(net, train_loader, loss, opt)
        dis_mean_pos, dis_mean_neg = test_one_epoch(net, test_loader)

        print('epoch{}:'.format(i),'loss:',train_loss)
        print('epoch{}:'.format(i),'mean distance between postive pair:',dis_mean_pos,'mean distance between negative pair:',dis_mean_neg)

        boardio.add_scalar('trainloss',train_loss,i)
        boardio.add_scalar('mean_distance_between_postive_pair',dis_mean_pos,i)
        boardio.add_scalar('mean_distance_between_negative_pair',dis_mean_neg,i)

        # if torch.cuda.device_count() > 1:
        #     torch.save(net.module.state_dict(), root + 'epoch{}.pkl'.format(i))
        # else:
        #     torch.save(net.state_dict(), root + 'epoch{}.pkl'.format(i))
        torch.save(net.state_dict(), 'checkpoints/' + 'epoch{}.pkl'.format(i))

    print('finish')




if __name__ == '__main__':

    train_loader = DataLoader(
        Synthetic_corr(partition='train', outlier_rate=0.98),
        batch_size= 16, shuffle=True, drop_last=True)
    test_loader = DataLoader(
        Synthetic_corr(partition='test', outlier_rate=0.98),
        batch_size=16, shuffle=False, drop_last=False)

    net = PointMI().cuda()
    loss = MICLoss(partten = 'standard')
    # net = torch.nn.DataParallel(net, device_ids=[0,1])
    # loss = torch.nn.DataParallel(loss, device_ids=[0,1])

    root = 'checkpoints/' + str(int(time.time()))
    boardio = SummaryWriter(log_dir= root)

    train(net, train_loader, test_loader, loss, boardio)

    boardio.close()