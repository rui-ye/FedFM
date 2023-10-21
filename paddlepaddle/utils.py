import os
import logging
import numpy as np
import paddle
import paddle.vision.transforms as transforms
import paddle.io as data
# from torch.autograd import Variable
import paddle.nn.functional as F
import paddle.nn as nn
import random
from sklearn.metrics import confusion_matrix
import sys

from model import *
from datasets import CIFAR10_truncated

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass



def load_cifar10_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = CIFAR10_truncated(root="/GPFS/rhome/xinyuzhu/.cache/paddle/dataset/cifar/cifar-10-python.tar.gz", train='train', download=True, transform=transform)
    cifar10_test_ds = CIFAR10_truncated(root="/GPFS/rhome/xinyuzhu/.cache/paddle/dataset/cifar/cifar-10-python.tar.gz", train='test', download=True, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)


def record_net_data_stats(y_train, net_dataidx_map, logdir):
    net_cls_counts_dict = {}
    net_cls_counts_npy = np.array([])
    num_classes = int(y_train.max()) + 1

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts_dict[net_i] = tmp
        tmp_npy = np.zeros(num_classes)
        for i in range(len(unq)):
            tmp_npy[unq[i]] = unq_cnt[i]
        net_cls_counts_npy = np.concatenate(
                        (net_cls_counts_npy, tmp_npy), axis=0)
    net_cls_counts_npy = np.reshape(net_cls_counts_npy, (-1,num_classes))


    data_list=[]
    for net_id, data in net_cls_counts_dict.items():
        n_total=0
        for class_id, n_data in data.items():
            n_total += n_data
        data_list.append(n_total)
    print('mean:', np.mean(data_list))
    print('std:', np.std(data_list))
    logger.info('Data statistics: %s' % str(net_cls_counts_dict))

    print(net_cls_counts_npy)
    return net_cls_counts_npy


def partition_data(dataset, datadir, logdir, partition, n_parties, beta=0.5):
    if dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)

    n_train = y_train.shape[0]

    if partition == "noniid-labeldir" or partition == "noniid":
        min_size = 0
        min_require_size = 10
        K = 10


        N = y_train.shape[0]
        net_dataidx_map = {}

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
                # if K == 2 and n_parties <= 10:
                #     if np.min(proportions) < 200:
                #         min_size = 0
                #         break

        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
            print(len(net_dataidx_map[j]))
        class_dis = np.zeros((n_parties, K))

        for j in range(n_parties):
            for m in range(K):
                class_dis[j,m] = (np.array(y_train[idx_batch[j]])==m).sum()
        print(class_dis.astype(int))


    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)
    return (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts)


# def save_features(model, dataloaders, save_dir, round):
#     was_training = False
#     if model.training:
#         model.eval()
#         was_training = True
#     for idx, dataloader in enumerate(dataloaders):
#         images, labels_idx = next(iter(dataloader))
#         images = paddle.to_tensor(images)
#         # labels = labels.cuda()
#         with paddle.no_grad():
#             representation_idx, _, _ = model(images)
#         representation_idx = representation_idx.cpu().numpy()
#         labels_idx = labels_idx.cpu().numpy()
#         if idx==0:
#             representations=representation_idx
#             labels=labels_idx
#         else:
#             representations = np.concatenate(
#                 (representations, representation_idx), axis=0)
#             labels = np.concatenate(
#                 (labels, labels_idx), axis=0)
#     np.save(os.path.join(save_dir, 'global_model_rep'+str(round)+'.npy'),representations)
#     np.save(os.path.join(save_dir, 'global_model_label'+str(round)+'.npy'),labels)
#
#    if was_training:
#        model.train()


def compute_accuracy(model, dataloader, get_confusion_matrix=False, device="cpu", multiloader=False):
    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    correct, total = 0, 0
    # if device == 'cpu':
    #     criterion = nn.CrossEntropyLoss()
    # elif "gpu" in device.type:
    #     criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss()
    loss_collector = []
    if multiloader:
        for loader in dataloader:
            with paddle.no_grad():
                for batch_idx, (x, target) in enumerate(loader):
                    #print("x:",x)
                    #print("target:",target)
                    if device != 'cpu':
                        x, target = paddle.to_tensor(x), paddle.to_tensor(target,dtype='int64')
                    _, _, out = model(x)
                    # print(out.shape, target.shape)
                    if len(target)==1:
                        out= out.unsqueeze(0)
                        loss = criterion(out, target)
                    else:
                        loss = criterion(out, target)
                    #_, pred_label = torch.max(out.data, 1)
                    pred_label = paddle.argmax(out,axis=1)
                    loss_collector.append(loss.item())
                    # total += x.data.size()[0]
                    total += x.shape[0]
                    correct += paddle.sum((pred_label == target)).item()

                    if device == "cpu":
                        pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                        true_labels_list = np.append(true_labels_list, target.numpy())
                    else:
                        pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                        true_labels_list = np.append(true_labels_list, target.numpy())
        avg_loss = sum(loss_collector) / len(loss_collector)
    else:
        with paddle.no_grad():
            for batch_idx, (x, target) in enumerate(dataloader):
                #print("x:",x)
                if device != 'cpu':
                     x, target = paddle.to_tensor(x), paddle.to_tensor(target,dtype='int64')
                _,_,out = model(x)
                loss = criterion(out, target)
                # _, pred_label = torch.max(out.data, 1)
                pred_label = paddle.argmax(out,axis=1)
                loss_collector.append(loss.item())
                # total += x.data.size()[0]
                total += x.shape[0]
                correct += paddle.sum((pred_label == target)).item()

                if device == "cpu":
                    pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                    true_labels_list = np.append(true_labels_list, target.numpy())
                else:
                    pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                    true_labels_list = np.append(true_labels_list, target.numpy())
            avg_loss = sum(loss_collector) / len(loss_collector)

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

    if was_training:
        model.train()

    if get_confusion_matrix:
        return correct / float(total), conf_matrix, avg_loss

    return correct / float(total), avg_loss

def compute_loss(model, dataloader, device="cpu"):
    was_training = False
    if model.training:
        model.eval()
        was_training = True
    if device == 'cpu':
        criterion = nn.CrossEntropyLoss()
    elif "gpu" in device.type:
        criterion = nn.CrossEntropyLoss().cuda()
    loss_collector = []
    with paddle.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):
            if device != 'cpu':
                x, target = paddle.to_tensor(x), paddle.to_tensor(target,dtype='int64')
            _,_,out = model(x)
            loss = criterion(out, target)
            loss_collector.append(loss.item())

        avg_loss = sum(loss_collector) / len(loss_collector)

    if was_training:
        model.train()

    return avg_loss


def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None, drop_last=True, noise_level=0):
    paddle.disable_static()
    if dataset in ('cifar10', 'cifar100'):
        if dataset == 'cifar10':
            dl_obj = CIFAR10_truncated

            normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                             std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
            # transform_train = transforms.Compose([
            #     transforms.ToTensor(),
            #     transforms.Lambda(lambda x: F.pad(
            #         Variable(x.unsqueeze(0), requires_grad=False),
            #         (4, 4, 4, 4), mode='reflect').data.squeeze()),
            #     transforms.ToPILImage(),
            #     transforms.ColorJitter(brightness=noise_level),
            #     transforms.RandomCrop(32),
            #     transforms.RandomHorizontalFlip(),
            #     transforms.ToTensor(),
            #     normalize
            # ])
            transform_train=transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
            # data prep for test set
            # transform_test = transforms.Compose([
            #     transforms.ToTensor(),
            #     normalize])
            transform_test=transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])



        train_ds = dl_obj(root="/GPFS/rhome/xinyuzhu/.cache/paddle/dataset/cifar/cifar-10-python.tar.gz", dataidxs=dataidxs, train='train', transform=transform_train, download=True)
        test_ds = dl_obj(root="/GPFS/rhome/xinyuzhu/.cache/paddle/dataset/cifar/cifar-10-python.tar.gz", train='test', transform=transform_test, download=True)
        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=drop_last, shuffle=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)

        return train_dl, test_dl, train_ds, test_ds
