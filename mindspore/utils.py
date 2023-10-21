import os
import logging
import numpy as np
import mindspore
import mindspore.dataset.vision as transforms
import mindspore.dataset as ds
import mindspore.dataset.vision
# from torch.autograd import Variable
# import torch.nn.functional as F
import mindspore.nn as nn
import random
from sklearn.metrics import confusion_matrix
import sys
import mindspore.dataset as data
from model import *
from datasets import CIFAR10_truncated, CIFAR100_truncated,Iterable
import matplotlib.pyplot as plt

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass



def load_cifar10_data(datadir):
    # transform = mindspore.dataset.transforms.Compose([transforms.ToTensor()])
    transform = None
    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    # ###### to see first 6 images ################
    # classes = []
    # with open(datadir + "/batches.meta.txt", "r") as f:
    #     for line in f:
    #         line = line.rstrip()
    #         if line:
    #             classes.append(line)
    # plt.figure()
    # for i in range(6):
    #     plt.subplot(2, 3, i + 1)
    #     image_trans = np.transpose(X_test[i], (1, 2, 0))
    #     mean = np.array([0.4914, 0.4822, 0.4465])
    #     std = np.array([0.2023, 0.1994, 0.2010])
    #     image_trans = std * image_trans + mean
    #     image_trans = np.clip(image_trans, 0, 1)
    #     plt.title(f"{classes[y_test[i]]}")
    #     plt.imshow(image_trans)
    #     plt.axis("off")
    #     plt.savefig("figure.png")
    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)

def load_cifar100_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar100_train_ds = CIFAR100_truncated(datadir, train=True, download=True, transform=transform)
    cifar100_test_ds = CIFAR100_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar100_train_ds.data, cifar100_train_ds.target
    X_test, y_test = cifar100_test_ds.data, cifar100_test_ds.target

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

def partition_data(dataset, datadir, logdir, partition, n_parties, beta=0.4, n_niid_parties=5):
    if dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    elif dataset == 'cifar100':
        X_train, y_train, X_test, y_test = load_cifar100_data(datadir)

    n_train = y_train.shape[0]

    if partition == "homo" or partition == "iid":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}

    elif partition == "noniid-labeldir" or partition == "noniid":
        min_size = 0
        min_require_size = 10
        K = 10
        if dataset == 'cifar100':
            K = 100
        elif dataset == 'tinyimagenet':
            K = 200
            # min_require_size = 100
        elif dataset == 'ham10000':
            K = 7

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
        print(class_dis)

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)
    return (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts)

def save_features(model, dataloaders, save_dir, round):
    was_training = False
    if model.training:
        model.eval()
        was_training = True
    for idx, dataloader in enumerate(dataloaders):
        images, labels_idx = next(iter(dataloader))
        images = images.cuda()
        # labels = labels.cuda()
        # with torch.no_grad():
        representation_idx, _, _ = model(images)
        representation_idx = representation_idx.cpu().numpy()
        labels_idx = labels_idx.cpu().numpy()
        if idx==0:
            representations=representation_idx
            labels=labels_idx
        else:
            representations = np.concatenate(
                (representations, representation_idx), axis=0)
            labels = np.concatenate(
                (labels, labels_idx), axis=0)
    np.save(os.path.join(save_dir, 'global_model_rep'+str(round)+'.npy'),representations)
    np.save(os.path.join(save_dir, 'global_model_label'+str(round)+'.npy'),labels)

    if was_training:
        model.train()

def compute_accuracy(model, dataloader, get_confusion_matrix=False, device="cpu", multiloader=False):
    # was_training = False
    # if model.training:
    #     model.set_train(False)
    #     was_training = True
    model.set_train(False)
    true_labels_list, pred_labels_list = np.array([]), np.array([])

    correct, total = 0, 0
    # if device == 'cpu':
    #     criterion = nn.CrossEntropyLoss()
    # elif "cuda" in device.type:
    #     criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss()
    loss_collector = []
    if multiloader:
        for loader in dataloader:
            # with torch.no_grad():
                for batch_idx, (x, target) in enumerate(loader):
                    #print("x:",x)
                    #print("target:",target)
                    # if device != 'cpu':
                    #     x, target = x.cuda(), target.to(dtype=mindspore.int64).cuda()
                    _, _, out = model(x)
                    # print(out.shape, target.shape)
                    if len(target)==1:
                        out= out.unsqueeze(0)
                        loss = criterion(out, target).asnumpy()
                    else:
                        loss = criterion(out, target).asnumpy()
                    pred_label, _ = mindspore.ops.ArgMaxWithValue(axis=1)(out)
                    # print(pred_label)
                    loss_collector.append(loss)
                    # total += x.data.size()[0]
                    total += x.shape[0]
                    correct += (pred_label == target).sum()

                    if device == "cpu":
                        pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                        true_labels_list = np.append(true_labels_list, target.numpy())
                    else:
                        pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                        true_labels_list = np.append(true_labels_list, target.numpy())
        avg_loss = sum(loss_collector) / len(loss_collector)
    else:
        # with torch.no_grad():
            for batch_idx, (x, target) in enumerate(dataloader):
                #print("x:",x)
                # if device != 'cpu':
                #     x, target = x.cuda(), target.to(dtype=mindspore.int64).cuda()
                _,_,out = model(x)
                loss = criterion(out, target).asnumpy()
                pred_label, _ = mindspore.ops.ArgMaxWithValue(axis=1)(out)
                # print(pred_label)
                loss_collector.append(loss)
                # total += x.data.size()[0]
                total += x.shape[0]
                correct += (pred_label == target).sum()

                if device == "cpu":
                    pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                    true_labels_list = np.append(true_labels_list, target.numpy())
                else:
                    pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                    true_labels_list = np.append(true_labels_list, target.numpy())
            avg_loss = sum(loss_collector) / len(loss_collector)

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

    # if was_training:
    #     model.train()
    model.set_train(True)

    if get_confusion_matrix:
        return correct / float(total), conf_matrix, avg_loss

    return correct / float(total), avg_loss

def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None, drop_last=True, noise_level=0):
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
            # transform_train=transforms.Compose(
            # [transforms.ToTensor(),
            #  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
            transform_train=[
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),transforms.HWC2CHW()]
                # transforms.Rescale(1.0 / 255.0, 0.0),
                
            # data prep for test set
            # transform_test = transforms.Compose([
            #     transforms.ToTensor(),
            #     normalize])
            # transform_test=transforms.Compose(
            # [transforms.ToTensor(),
            #  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
            transform_test=[
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),transforms.HWC2CHW()]
                # transforms.Rescale(1.0 / 255.0, 0.0),
                

        elif dataset == 'cifar100':
            dl_obj = CIFAR100_truncated

            normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                             std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
            # transform_train = transforms.Compose([
            #     transforms.RandomCrop(32),
            #     transforms.RandomHorizontalFlip(),
            #     transforms.ToTensor(),
            #     normalize
            # ])
            transform_train = transforms.Compose([
                # transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                normalize
            ])
            # data prep for test set
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                normalize])


        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
        test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)
        # train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=drop_last, shuffle=True)
        # test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)


        # train = Iterable(train_ds.data,train_ds.target)
        # test = Iterable(test_ds.data,test_ds.target)
        # train_dataloader = mindspore.dataset.GeneratorDataset(source=train,column_names=["data", "label"])
        # train_dataloader.batch(batch_size=train_bs,drop_remainder=drop_last)
        # train_dataloader.shuffle(buffer_size=10)
        # train_dataloader.create_dict_iterator()
        # test_dataloader = mindspore.dataset.GeneratorDataset(source=test, column_names=["data","label"])
        # test_dataloader.batch(batch_size=test_bs)
        # test_dataloader.create_dict_iterator()

        # train_dl = train_dl.batch(batch_size=train_bs,drop_remainder=drop_last)
        # train_dl = train_dl.shuffle()
        # train_dataloader = train_dl.create_dict_iterator()
        # test_dl = test_ds
        # test_dl.batch(batch_size=test_bs)
        # test_dataloader = test_dl.create_dict_iterator()
        train_dataloader = mindspore.dataset.GeneratorDataset(train_ds, column_names=["data","label"],shuffle= True)
        train_dataloader=train_dataloader.batch(train_bs,drop_remainder=drop_last)
        test_dataloader = mindspore.dataset.GeneratorDataset(test_ds, column_names=["data","label"],shuffle=False)
        test_dataloader=test_dataloader.batch(test_bs)


        return train_dataloader, test_dataloader, train_ds, test_ds

if __name__ == '__main__':
    mindspore.set_context(device_target='GPU',device_id=0)
