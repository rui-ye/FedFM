# scene2
# fedfm val acc:0.405838 , test acc:0.400200
# fedavg val acc:0.374050 , test acc:0.367400
import numpy as np
import json
import paddle
import paddle.nn.functional as F
import paddle.nn
import paddle.optimizer as optim
import paddle.nn as nn
import argparse
import logging
import os
import copy
import datetime
import random
import math

from prepare_data import get_dataloader_new
from model import ModelFedCon_noheader
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
    parser.add_argument('--epochs', type=int, default=1, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=10, help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='fedfm',
                        help='communication strategy: fedavg/fedprox')
    parser.add_argument('--comm_round', type=int, default=100, help='number of maximum communication roun')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='gpu:7', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox or moon')
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    parser.add_argument('--local_max_epoch', type=int, default=100, help='the number of epoch for local optimal training')
    parser.add_argument('--model_buffer_size', type=int, default=1, help='store how many previous models for contrastive loss')
    parser.add_argument('--pool_option', type=str, default='FIFO', help='FIFO or BOX')
    parser.add_argument('--sample_fraction', type=float, default=1.0, help='how many clients are sampled in each round')
    parser.add_argument('--load_model_file', type=str, default=None, help='the model to load as global model')
    parser.add_argument('--load_pool_file', type=str, default=None, help='the old model pool path to load')
    parser.add_argument('--load_model_round', type=int, default=None, help='how many rounds have executed for the loaded model')
    parser.add_argument('--load_first_net', type=int, default=1, help='whether load the first net as old net or not')
    parser.add_argument('--normal_model', type=int, default=0, help='use normal model or aggregate model')
    parser.add_argument('--loss', type=str, default='contrastive')
    parser.add_argument('--save_model',type=int,default=0)
    parser.add_argument('--use_project_head', type=int, default=0)
    parser.add_argument('--server_momentum', type=float, default=0, help='the server momentum (FedAvgM)')
    parser.add_argument('--skew_class', type=int, default = 2, help='The parameter for the noniid-skew for data partitioning')

    parser.add_argument('--print_local_test_acc', type=int, default=1)
    parser.add_argument('--save_feature', type=int, default=0)

    parser.add_argument('--lam_fm', type=float, default=50.0)
    parser.add_argument('--start_ep_fm', type=int, default=20, help='which round to start fm')
    parser.add_argument('--l2match', action='store_true')
    parser.add_argument('--fm_avg_anchor', type=int, default=0, help='equally average')
    parser.add_argument('--cg_tau', type=float, default=0.1, help='tempreture')


    args = parser.parse_args()
    return args


def init_nets(net_configs, n_parties, args, device='cpu'):
    nets = {net_i: None for net_i in range(n_parties)}
    if args.dataset in {'mnist', 'cifar10', 'svhn', 'fmnist', 'cinic10', 'cinic10_val'}:
        n_classes=10

    for net_i in range(n_parties):
        net = ModelFedCon_noheader(args.model, args.out_dim, n_classes, net_configs)
        net.to(args.device)
        nets[net_i] = net

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)
    return nets, model_meta_data, layer_type


def train_net_fm(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, global_centroids, start_fm, device="cpu"):
    net.to(args.device)
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    # only support SGDMomentum in paddle version now
    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.Momentum(parameters=net.parameters(), learning_rate=args.lr, momentum=0.9,weight_decay=paddle.regularizer.L2Decay(args.reg))
        # optimizer = optim.Momentum(parameters=filter(lambda p:p.grad!=None,net.parameters()), learning_rate=args.lr, momentum=0.9,weight_decay=args.reg)
    criterion = paddle.nn.CrossEntropyLoss()

    cnt = 0

    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = paddle.to_tensor(x), paddle.to_tensor(target,dtype='int64')
           
            x.stop_gradient = True
            target.stop_gradient = True
            optimizer.clear_grad()

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

    logger.info(' ** Training complete **')
    return test_acc


def local_train_net(nets, args, net_dataidx_map, train_dl=None, test_dl=None, global_model = None, prev_model_pool = None, server_c = None, clients_c = None, round=None, device="cpu", current_round=0):
    avg_acc = 0.0
    acc_list = []

    for net_id, net in nets.items():
        dataidxs = net_dataidx_map[net_id]
        for name, param in net.named_parameters():
            if not param.stop_gradient:
                print(name)
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
    return nets

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
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(args.dataset, args.datadir, dataset_logdir, args.partition, args.n_parties, beta=args.beta)
    print('successfully complete fuction: partition_data')
    n_party_per_round = int(args.n_parties * args.sample_fraction)
    party_list = [i for i in range(args.n_parties)]
    party_list_rounds = []
    if n_party_per_round != args.n_parties:
        for i in range(args.comm_round):
            party_list_rounds.append(random.sample(party_list, n_party_per_round))
    else:
        for i in range(args.comm_round):
            party_list_rounds.append(party_list)

    print('will start fuction: get_dataloader')
    if args.dataset != 'cinic10_val':
        train_dl_global, test_dl, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                                args.datadir,
                                                                                args.batch_size,
                                                                                args.batch_size)

    print('successfully complete fuction:get_dataloader')
    print("len train_dl_global:", len(train_ds_global))
    train_dl=None
    data_size = len(test_ds_global)
    print("len test_ds_global:", len(test_ds_global))
    print('initializing nets')
    logger.info("Initializing nets")
    nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.n_parties, args, device='cpu')

    global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 1, args, device='cpu')
    global_model = global_models[0]
    n_comm_rounds = args.comm_round

    print(device)
    if args.alg=='fedfm' or args.alg=='fedproc':
        if args.dataset in {'mnist', 'cifar10', 'svhn', 'fmnist', 'cinic10', 'cinic10_val'}:
            n_classes=10
        if args.model.startswith('resnet18'):
            global_centroids = paddle.zeros((n_classes, 512))
        elif args.model.startswith('resnet50'):
            global_centroids = paddle.zeros((n_classes, 2048))
        global_centroids = paddle.to_tensor(global_centroids)
    train_local_dls=[]    
    val_local_dls=[]
    print('will start preparing train_dl_local')
    if args.dataset!='cinic10_val':
        for net_id, net in nets.items():
            dataidxs = net_dataidx_map[net_id]
            dataidxs_t = dataidxs[:int(0.8*len(dataidxs))]
            dataidxs_v = dataidxs[int(0.8*len(dataidxs)):]
            train_dl_local, _, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_t)
            train_local_dls.append(train_dl_local)
            print('successfully complete one train_dl_local')
            if args.save_feature:
                val_dl_local, _, _, _ = get_dataloader(args.dataset, args.datadir, 200, 32, dataidxs_v, drop_last=False)
            else:
                val_dl_local, _, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_v, drop_last=False)
                print('successfully complete one val_dl_local')
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
            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())

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

            logger.info('global n_test: %d' % len(test_dl))
            global_model.to(args.device)


            #feature_matching
            local_centroids, local_distributions = get_client_centroids_info(global_model, dataloaders=train_local_dls, model_name=args.model, dataset_name=args.dataset, party_list_this_round=party_list_this_round)
            global_centroids = get_global_centroids(local_centroids, local_distributions, global_centroids, momentum=0.0, equally_average=args.fm_avg_anchor)


            val_acc, _ = compute_accuracy(global_model, val_local_dls, device=device, multiloader=True)
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)
            test_acc_list.append(test_acc)

            if(best_acc<val_acc):
                best_acc=val_acc
                best_test_acc=test_acc
                logger.info('New Best val acc:%f , test acc:%f'%(val_acc,test_acc))
            else:
                logger.info('>> Global Model Train accuracy: %f' % val_acc)
                logger.info('>> Global Model Test accuracy: %f' % test_acc)
                logger.info('>> Global Model Best accuracy: %f' % best_test_acc)
            
            print(f'>> Round {round} test accuracy : {test_acc} | Best Acc : {best_test_acc}')
           
