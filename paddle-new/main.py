# from importlib_metadata import distribution
import numpy as np
import json
import paddle
import paddle.optimizer as optim
import paddle.nn as nn
import argparse
import logging
import os
import copy
import datetime
import random
import math


from model import *
from utils import *

from feature_matching.utils import get_client_centroids_info, get_global_centroids
from feature_matching.loss_f import matching_cross_entropy, matching_l2



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet18_7', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset used for training')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='noniid', help='the data partitioning strategy')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.1)')
    parser.add_argument('--epochs', type=int, default=10, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=10, help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='fedfm',
                        help='communication strategy: fedavg/fedprox')
    parser.add_argument('--comm_round', type=int, default=100, help='number of maximum communication roun')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/cifar-10-python.tar.gz", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cuda:4', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    parser.add_argument('--sample_fraction', type=float, default=1.0, help='how many clients are sampled in each round')
    parser.add_argument('--print_local_test_acc', type=int, default=1)
    parser.add_argument('--save_feature', type=int, default=0)
    #fedFM parameters
    parser.add_argument('--lam_fm', type=float, default=50.0)
    parser.add_argument('--start_ep_fm', type=int, default=20, help='which round to start fm')
    parser.add_argument('--l2match', action='store_true')
    parser.add_argument('--fm_avg_anchor', type=int, default=0, help='equally average')
    parser.add_argument('--cg_tau', type=float, default=0.1, help='tempreture')
    parser.add_argument('--n_niid_parties', type=int, default=5, help='number of niid workers')

    args = parser.parse_args()
    return args


def init_nets(net_configs, n_parties, args, device='cpu'):
    nets = {net_i: None for net_i in range(n_parties)}
    if args.dataset in {'mnist', 'cifar10', 'svhn', 'fmnist', 'cinic10', 'cinic10_val'}:
        n_classes = 10
    elif args.dataset == 'cifar100':
        n_classes = 100

    for net_i in range(n_parties):
        net = ModelFedCon_noheader(args.model, args.out_dim, n_classes, net_configs)
        if device == 'cpu':
            net.to(device)
        else:
            net = net.to(args.device)
        nets[net_i] = net

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)
    return nets, model_meta_data, layer_type

def local_train_net(nets, args, net_dataidx_map, train_dl=None, test_dl=None, global_model = None, prev_model_pool = None, server_c = None, clients_c = None, round=None, device="cpu", current_round=0):
    avg_acc = 0.0
    acc_list = []
    if global_model:
        global_model.to(args.device)
    for net_id, net in nets.items():
        dataidxs = net_dataidx_map[net_id]
        # for name, param in net.named_parameters():
        #     if not param.requires_grad:
        #         print(name)
        # print(net.device)

        # print("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        train_dl_local=train_dl[net_id]
        # train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs
        if args.alg =='fedfm':
            testacc = train_net_fm(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args, global_centroids, current_round>=args.start_ep_fm,
                                        device=device)

        logger.info("net %d final test acc %f" % (net_id, testacc))
        print("net %d, n_training: %d, final test acc %f" % (net_id, len(dataidxs), testacc))
        avg_acc += testacc
        acc_list.append(testacc)
    avg_acc /= args.n_parties
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)
        logger.info("std acc %f" % np.std(acc_list))
    if global_model:
        global_model.to('cpu')
    return nets

def train_net_fm(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, global_centroids, start_fm, device="cpu"):
    # net = nn.DataParallel(net)
    net.to(args.device)
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        # optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), learning_rate=lr, momentum=0.9,
        #                       weight_decay=args.reg)
        optimizer = optim.Momentum(parameters=net.parameters(), learning_rate=args.lr, momentum=0.9,
            weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss()

    cnt = 0

    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.cuda(), target.cuda()

            optimizer.clear_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            representation,_,out = net(x)

            if start_fm:
                if args.l2match:
                    loss_fm = args.lam_fm * matching_l2(features=representation, labels=target, centroids=global_centroids)
                else:

                    loss_fm = args.lam_fm * matching_cross_entropy(representation, labels=target,
                                                centroids=global_centroids, tao=args.cg_tau)
            else:
                loss_fm = 0.0
            loss_1 = criterion(out, target)
            loss = loss_1+loss_fm
                
            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))
    
    if args.print_local_test_acc:
        # train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
        test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
        # logger.info('>> Training accuracy: %f' % train_acc)
        logger.info('>> Test accuracy: %f' % test_acc)
    else:
        test_acc = 0.0

    net.to('cpu')

    logger.info(' ** Training complete **')
    return test_acc


if __name__ == '__main__':
    args = get_args()
    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    now_time = datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
    print(now_time)
    print(args)

    dataset_logdir = os.path.join(args.logdir, args.dataset)
    mkdirs(dataset_logdir)

    if args.log_file_name is None:
        argument_path = 'experiment_arguments-%s.json' % (now_time)
    else:
        argument_path = args.log_file_name + '.json'
    with open(os.path.join(dataset_logdir, argument_path), 'w') as f:
        json.dump(str(args), f)
    device = paddle.set_device(args.device)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if args.log_file_name is None:
        args.log_file_name = 'experiment_log-%s' % (now_time)
    log_path = args.log_file_name + '.log'
    logging.basicConfig(
        filename=os.path.join(dataset_logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info(device)

    seed = args.init_seed
    logger.info("#" * 100)
    np.random.seed(seed)
    paddle.seed(seed)
    random.seed(seed)

    logger.info("Partitioning data")


    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        args.dataset, args.datadir, dataset_logdir, args.partition, args.n_parties, beta=args.beta, n_niid_parties=args.n_niid_parties)

    n_party_per_round = int(args.n_parties * args.sample_fraction)
    party_list = [i for i in range(args.n_parties)]
    party_list_rounds = []
    if n_party_per_round != args.n_parties:
        for i in range(args.comm_round):
            party_list_rounds.append(random.sample(party_list, n_party_per_round))
    else:
        for i in range(args.comm_round):
            party_list_rounds.append(party_list)

    n_classes = len(np.unique(y_train))

    if args.dataset != 'cinic10_val':
        train_dl_global, test_dl, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                                args.datadir,
                                                                                args.batch_size,
                                                                                args.batch_size)


    print("len train_dl_global:", len(train_ds_global))
    train_dl=None
    data_size = len(test_ds_global)

    logger.info("Initializing nets")
    nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.n_parties, args, device='cpu')

    global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 1, args, device='cpu')
    global_model = global_models[0]
    n_comm_rounds = args.comm_round

    print(device)
    if args.alg=='fedfm':
        if args.dataset in {'mnist', 'cifar10', 'svhn', 'fmnist', 'cinic10', 'cinic10_val'}:
            n_classes = 10
        elif args.dataset == 'cifar100':
            n_classes = 100

        if args.model.startswith('resnet18'):
            global_centroids = paddle.zeros((n_classes, 512))
        elif args.model.startswith('resnet50'):
            global_centroids = paddle.zeros((n_classes, 2048))
        global_centroids = global_centroids.cuda()
    train_local_dls=[]    
    val_local_dls=[]
    if args.dataset!='cinic10_val':
        for net_id, net in nets.items():
            print(f"net id {net_id}, dataloader prepared")
            dataidxs = net_dataidx_map[net_id]
            dataidxs_t = dataidxs[:int(0.8*len(dataidxs))]
            dataidxs_v = dataidxs[int(0.8*len(dataidxs)):]
            train_dl_local, _, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_t)
            train_local_dls.append(train_dl_local)
            if args.save_feature:
                val_dl_local, _, _, _ = get_dataloader(args.dataset, args.datadir, 200, 32, dataidxs_v, drop_last=False)
            else:
                val_dl_local, _, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_v, drop_last=False)
            val_local_dls.append(val_dl_local)


    best_acc=0
    best_test_acc=0
    test_acc_list = []
    acc_dir = os.path.join(dataset_logdir, 'acc_list')
    if not os.path.exists(acc_dir):
        os.mkdir(acc_dir)
    feature_dir = os.path.join(dataset_logdir, 'features')
    if not os.path.exists(feature_dir):
        os.mkdir(feature_dir)
    feature_dir = os.path.join(feature_dir, f'{now_time}')
    if not os.path.exists(feature_dir):
        os.mkdir(feature_dir)

    acc_path = os.path.join(dataset_logdir, f'acc_list/{now_time}.npy')

    if args.alg == 'fedfm':
        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]

            global_w = global_model.state_dict()

            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.set_state_dict(global_w)
            
            local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_local_dls, test_dl=test_dl, device=device, current_round=round)

            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]

            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]


            global_model.set_state_dict(global_w)

            #logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl))
            global_model.to(args.device)
            # train_acc, train_loss = compute_accuracy(global_model, train_local_dls, device=device, multiloader=True)

            #feature_matching
            global_centroids = global_centroids.cpu()
            local_centroids, local_distributions = get_client_centroids_info(global_model, dataloaders=train_local_dls, model_name=args.model, dataset_name=args.dataset, party_list_this_round=party_list_this_round)
            global_centroids = get_global_centroids(local_centroids, local_distributions, global_centroids, momentum=0.0, equally_average=args.fm_avg_anchor)
            global_centroids = global_centroids.cuda()

            if args.save_feature and (round+1)%10==0:
                save_features(model=global_model, dataloaders=val_local_dls, save_dir=feature_dir, round=round)
            val_acc, _ = compute_accuracy(global_model, val_local_dls, device=device, multiloader=True)
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)
            test_acc_list.append(test_acc)

            if(best_acc<val_acc):
                best_acc=val_acc
                best_test_acc=test_acc
                logger.info('New Best val acc:%f , test acc:%f'%(val_acc,test_acc))
            else:
                # logger.info('>> Global Model Train accuracy: %f' % train_acc)
                logger.info('>> Global Model Train accuracy: %f' % val_acc)
                logger.info('>> Global Model Test accuracy: %f' % test_acc)
                logger.info('>> Global Model Best accuracy: %f' % best_test_acc)
            
            print(f'>> Round {round} test accuracy : {test_acc} | Best Acc : {best_test_acc}')
           
            mkdirs(args.modeldir+'fedfm/')
            global_model.to('cpu')

            paddle.save(global_model.state_dict(), args.modeldir+'fedfm/'+'globalmodel'+args.log_file_name+'.pth')
            paddle.save(nets[0].state_dict(), args.modeldir+'fedfm/'+'localmodel0'+args.log_file_name+'.pth')
    

    
    print('>> Global Model Best accuracy: %f' % best_test_acc)
    print(args)
    print(f'>> Start time : {now_time} | End time : {datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")}')
    logger.info('>> Test ACC List: %s' % str(test_acc_list))
    np.save(acc_path, np.array(test_acc_list))