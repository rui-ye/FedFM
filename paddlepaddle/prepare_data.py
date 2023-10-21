import os
import logging
import numpy as np
import paddle
import paddle.vision.transforms as transforms
import paddle.io as data
# from datasets import CIFAR10_truncated, CIFAR100_truncated, ImageFolder_custom, ImageFolder_WIKI, FashionMNIST_truncated, MNIST_truncated, digit5_dataset_read, rotated_cifar10_dataset_read, get_domainnet_dloader, cache_domainnet, get_domainnet10_dloader
from cvdataset import cifar_dataset_read
# from nlpdataset import nlpdataset_read

def get_dataloader_new(args,logger):
    if args.dataset in ('cifar10', 'cifar100'):
        train_dataloaders, val_dataloaders, test_loader, net_dataidx_map, traindata_cls_counts, data_distributions = cifar_dataset_read(args.dataset, args.datadir, args.batch_size, args.n_parties, args.partition, args.beta, args.skew_class,logger)
    elif args.dataset in ('yahoo_answers'):
        #    train_dataloaders, val_dataloaders, test_loader, net_dataidx_map, traindata_cls_counts, data_distributions = nlpdataset_read(args.dataset, args.datadir, args.batch_size, args.n_parties, args.partition, args.beta, args.skew_class)
        pass
    return train_dataloaders, val_dataloaders, test_loader, net_dataidx_map, traindata_cls_counts, data_distributions