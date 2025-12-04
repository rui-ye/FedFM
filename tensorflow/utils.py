import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
from sklearn.metrics import confusion_matrix
import sys

from model import *
from datasets import CIFAR10_truncated, CIFAR100_truncated

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass


def normalize_image(img, mean, std):
    """Normalize image tensor"""
    mean = tf.constant(mean, dtype=tf.float32)
    std = tf.constant(std, dtype=tf.float32)
    return (img - mean) / std


def load_cifar10_data(datadir):
    """Load CIFAR-10 data from local batch files - loads full training set (all 5 batches)"""
    # Try multiple possible base paths
    possible_base_paths = [
        datadir,
        os.path.join(datadir, 'cifar-10-batches-py'),
        './data/cifar-10-batches-py',
    ]
    
    base_path = None
    for path in possible_base_paths:
        test_file = os.path.join(path, 'test_batch')
        if os.path.exists(test_file):
            base_path = path
            break
    
    if base_path is None:
        # Fallback: use directory path
        base_path = datadir
    
    # Load full training set (all 5 batches)
    cifar10_train_ds = CIFAR10_truncated(base_path, train=True, download=False, transform=None)
    
    # Load test set
    cifar10_test_ds = CIFAR10_truncated(base_path, train=False, download=False, transform=None)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target
    
    # Convert to numpy if needed
    if isinstance(X_train, tf.Tensor):
        X_train = X_train.numpy()
    if isinstance(y_train, tf.Tensor):
        y_train = y_train.numpy()
    if isinstance(X_test, tf.Tensor):
        X_test = X_test.numpy()
    if isinstance(y_test, tf.Tensor):
        y_test = y_test.numpy()
    
    print(f"Loaded CIFAR-10: Train shape {X_train.shape} ({len(X_train)} samples), Test shape {X_test.shape} ({len(X_test)} samples)")
    return (X_train, y_train, X_test, y_test)


def load_cifar100_data(datadir):
    """Load CIFAR-100 data"""
    # Similar implementation for CIFAR-100 if needed
    raise NotImplementedError("CIFAR-100 loading not implemented yet")


def record_net_data_stats(y_train, net_dataidx_map, logdir):
    """Record data statistics for each client"""
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
    """Partition data for federated learning"""
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

        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
            print(len(net_dataidx_map[j]))
        class_dis = np.zeros((n_parties, K))

        for j in range(n_parties):
            for m in range(K):
                class_dis[j,m] = (np.array(y_train[idx_batch[j]])==m).sum()
        print(class_dis)

    elif partition=='noniid-2' and dataset=='cifar10':
        num_per_shard = 100
        num_shards = n_train//num_per_shard
        idx_shard = [i for i in range(num_shards)]
        net_dataidx_map = {i: np.array([]).astype(int) for i in range(n_parties)}
        idxs = np.arange(num_shards*num_per_shard).astype(int)
        labels = y_train

        print(n_train)
        print(labels.max()+1)

        idxs_labels = np.vstack((idxs, labels)).astype(int)
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]

        if n_parties==10:
            num_shard_per_class = num_shards//(labels.max()+1)
            print('User_class_distribution:')
            for i in range(n_parties):
                small_shards = 3
                chosen_set = set(range(i*num_shard_per_class,(i+1)*num_shard_per_class-10*small_shards))
                for j in range(10):
                    for q in range(small_shards):
                        chosen_set.add((j+1)*num_shard_per_class-10*small_shards+i*small_shards+q)

                class_dict = {}
                for j in range(10):
                    class_dict[j] = 0

                for chosen in chosen_set:
                    net_dataidx_map[i] = np.concatenate(
                        (net_dataidx_map[i], idxs[chosen*num_per_shard:(chosen+1)*num_per_shard]), axis=0)
                    class_dict[idxs_labels[1,chosen*num_per_shard]] += 1

                np.random.shuffle(net_dataidx_map[i])
                net_dataidx_map[i] = list(net_dataidx_map[i])
                print(class_dict)

    elif partition=='noniid-4' and dataset=='cifar10':
        labels = y_train
        
        num_non_iid_client = n_niid_parties
        num_iid_client =  n_parties-num_non_iid_client
        num_classes = int(labels.max()+1)
        num_sample_per_client = n_train//n_parties
        num_sample_per_class = n_train//num_classes
        num_per_shard = int(n_train/num_classes/(num_non_iid_client+num_iid_client))

        net_dataidx_map = {i: np.array([]).astype(int) for i in range(n_parties)}
        idxs = np.arange(n_train).astype(int)

        idxs_labels = np.vstack((idxs, labels)).astype(int)
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]

        for i in range(num_non_iid_client):
            net_dataidx_map[i] = np.concatenate(
                    (net_dataidx_map[i], idxs[((2*i)%10)*num_sample_per_class+num_per_shard*(i//5)*5:((2*i)%10)*num_sample_per_class+num_per_shard*(i//5+1)*5]), axis=0)
            print(((2*i)%10)*num_sample_per_class+num_per_shard*(i//5)*5,((2*i)%10)*num_sample_per_class+num_per_shard*(i//5+1)*5)
            net_dataidx_map[i] = np.concatenate(
                    (net_dataidx_map[i], idxs[((2*i+1)%10)*num_sample_per_class+num_per_shard*(i//5)*5:((2*i+1)%10)*num_sample_per_class+num_per_shard*(i//5+1)*5]), axis=0)
            print(((2*i+1)%10)*num_sample_per_class+num_per_shard*(i//5)*5,((2*i+1)%10)*num_sample_per_class+num_per_shard*(i//5+1)*5)
            np.random.shuffle(net_dataidx_map[i])
            net_dataidx_map[i] = list(net_dataidx_map[i])
        
        for i in range(num_non_iid_client,n_parties):
            for j in range(num_classes):
                net_dataidx_map[i] = np.concatenate(
                        (net_dataidx_map[i], idxs[j*num_sample_per_class+num_per_shard*5*(num_non_iid_client//5)+num_per_shard*(i-num_non_iid_client): \
                                                    j*num_sample_per_class+num_per_shard*5*(num_non_iid_client//5)+num_per_shard*(i-num_non_iid_client+1)]), axis=0)
                print(j*num_sample_per_class+num_per_shard*5*(num_non_iid_client//5)+num_per_shard*(i-num_non_iid_client),j*num_sample_per_class+num_per_shard*5*(num_non_iid_client//5)+num_per_shard*(i-num_non_iid_client+1))
            np.random.shuffle(net_dataidx_map[i])
            net_dataidx_map[i] = list(net_dataidx_map[i])
    
    elif partition=='noniid-5' and dataset=='cifar10':
        labels = y_train
        
        num_non_iid_client = n_niid_parties
        num_iid_client =  n_parties-num_non_iid_client
        num_classes = int(labels.max()+1)
        num_sample_per_client = 1000
        num_sample_per_class = n_train//num_classes
        
        net_dataidx_map = {i: np.array([]).astype(int) for i in range(n_parties)}
        idxs = np.arange(n_train).astype(int)

        idxs_labels = np.vstack((idxs, labels)).astype(int)
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]

        num_per_shard_niid = num_sample_per_client//2
        for i in range(num_non_iid_client):
            net_dataidx_map[i] = np.concatenate(
                    (net_dataidx_map[i], idxs[((2*i)%10)*num_sample_per_class+num_per_shard_niid*(i//5):((2*i)%10)*num_sample_per_class+num_per_shard_niid*(i//5+1)]), axis=0)
            print(((2*i)%10)*num_sample_per_class+num_per_shard_niid*(i//5),((2*i)%10)*num_sample_per_class+num_per_shard_niid*(i//5+1))
            net_dataidx_map[i] = np.concatenate(
                    (net_dataidx_map[i], idxs[((2*i+1)%10)*num_sample_per_class+num_per_shard_niid*(i//5):((2*i+1)%10)*num_sample_per_class+num_per_shard_niid*(i//5+1)]), axis=0)
            print(((2*i+1)%10)*num_sample_per_class+num_per_shard_niid*(i//5),((2*i+1)%10)*num_sample_per_class+num_per_shard_niid*(i//5+1))
            np.random.shuffle(net_dataidx_map[i])
            net_dataidx_map[i] = list(net_dataidx_map[i])
        
        num_per_shard_iid = num_sample_per_client//10
        for i in range(num_non_iid_client,n_parties):
            for j in range(num_classes):
                net_dataidx_map[i] = np.concatenate(
                        (net_dataidx_map[i], idxs[j*num_sample_per_class+num_per_shard_niid*(num_non_iid_client//5)+num_per_shard_iid*(i-num_non_iid_client): \
                                                    j*num_sample_per_class+num_per_shard_niid*(num_non_iid_client//5)+num_per_shard_iid*(i-num_non_iid_client+1)]), axis=0)
                print(j*num_sample_per_class+num_per_shard_niid*(num_non_iid_client//5)+num_per_shard_iid*(i-num_non_iid_client),j*num_sample_per_class+num_per_shard_niid*(num_non_iid_client//5)+num_per_shard_iid*(i-num_non_iid_client+1))
            np.random.shuffle(net_dataidx_map[i])
            net_dataidx_map[i] = list(net_dataidx_map[i])
    
    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)
    return (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts)


def save_features(model, dataloaders, save_dir, round):
    """Save features from model"""
    was_training = model.trainable
    model.trainable = False
    
    representations_list = []
    labels_list = []
    
    for idx, dataset in enumerate(dataloaders):
        for images, labels_idx in dataset:
            representation_idx, _, _ = model(images, training=False)
            representations_list.append(representation_idx.numpy())
            labels_list.append(labels_idx.numpy())
            break  # Only take first batch
    
    representations = np.concatenate(representations_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    
    np.save(os.path.join(save_dir, 'global_model_rep'+str(round)+'.npy'), representations)
    np.save(os.path.join(save_dir, 'global_model_label'+str(round)+'.npy'), labels)

    model.trainable = was_training


def compute_accuracy(model, dataloader, get_confusion_matrix=False, device="cpu", multiloader=False):
    """Compute accuracy of model on dataloader"""
    was_training = model.trainable
    model.trainable = False

    # Determine device string for TensorFlow
    if 'cuda' in device.lower() or 'gpu' in device.lower():
        if ':' in device:
            gpu_id = int(device.split(':')[1])
            tf_device = f'/GPU:{gpu_id}'
        else:
            tf_device = '/GPU:0'
    else:
        tf_device = '/CPU:0'

    true_labels_list, pred_labels_list = np.array([]), np.array([])
    correct, total = 0, 0
    loss_collector = []
    
    criterion = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    if multiloader:
        for loader in dataloader:
            for batch_idx, (x, target) in enumerate(loader):
                x = tf.cast(x, tf.float32)
                target = tf.cast(target, tf.int64)
                
                # With soft device placement, TensorFlow will use standard CUDA if CuDNN fails
                _, _, out = model(x, training=False)
                
                # Compute loss
                if len(target.shape) == 0:
                    target = tf.expand_dims(target, 0)
                    out = tf.expand_dims(out, 0)
                loss = criterion(target, out)
                loss_collector.append(loss.numpy())
                
                pred_label = tf.argmax(out, axis=1)
                total += x.shape[0]
                correct += tf.reduce_sum(tf.cast(pred_label == target, tf.int32)).numpy()
                
                pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                true_labels_list = np.append(true_labels_list, target.numpy())
        avg_loss = sum(loss_collector) / len(loss_collector) if loss_collector else 0.0
    else:
        for batch_idx, (x, target) in enumerate(dataloader):
            x = tf.cast(x, tf.float32)
            target = tf.cast(target, tf.int64)
            
            # With soft device placement, TensorFlow will use standard CUDA if CuDNN fails
            _, _, out = model(x, training=False)
            
            loss = criterion(target, out)
            loss_collector.append(loss.numpy())
            
            pred_label = tf.argmax(out, axis=1)
            total += x.shape[0]
            correct += tf.reduce_sum(tf.cast(pred_label == target, tf.int32)).numpy()
            
            pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
            true_labels_list = np.append(true_labels_list, target.numpy())
        avg_loss = sum(loss_collector) / len(loss_collector) if loss_collector else 0.0

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

    model.trainable = was_training

    if get_confusion_matrix:
        return correct / float(total), conf_matrix, avg_loss

    return correct / float(total), avg_loss


def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None, drop_last=True, noise_level=0):
    """Get TensorFlow dataset loaders"""
    if dataset in ('cifar10', 'cifar100'):
        if dataset == 'cifar10':
            # Use local path - try multiple possibilities
            possible_paths = [
                os.path.join(datadir, 'cifar-10-batches-py', 'data_batch_1'),
                os.path.join(datadir, 'data_batch_1'),
                './data/cifar-10-batches-py/data_batch_1',
            ]
            local_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    local_path = path
                    break
            if local_path is None:
                local_path = datadir
            
            # Load data without transform (we'll normalize in numpy)
            train_ds_obj = CIFAR10_truncated(local_path, dataidxs=dataidxs, train=True, transform=None, download=False)
            
            # Find test path
            possible_test_paths = [
                os.path.join(os.path.dirname(local_path), 'test_batch'),
                os.path.join(datadir, 'cifar-10-batches-py', 'test_batch'),
                os.path.join(datadir, 'test_batch'),
                './data/cifar-10-batches-py/test_batch',
            ]
            test_path = None
            for path in possible_test_paths:
                if os.path.exists(path):
                    test_path = path
                    break
            if test_path is None:
                test_path = datadir
            
            test_ds_obj = CIFAR10_truncated(test_path, train=False, transform=None, download=False)
            
            # Normalization parameters
            mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
            std = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)
            
            # Create TensorFlow datasets
            # Process all data in numpy first to avoid GPU issues during data loading
            # Use CPU for all data operations without modifying global GPU settings
            def create_dataset(ds_obj, shuffle_data=True):
                # Get raw data (already loaded in CIFAR10_truncated)
                images_np = ds_obj.data.astype(np.float32) / 255.0
                labels_np = ds_obj.target.astype(np.int64)
                
                # Apply normalization using numpy (broadcast across channels)
                # images_np shape: (N, 32, 32, 3)
                images_np = (images_np - mean) / std
                
                # Create dataset from numpy arrays
                # Use with_options to force CPU placement
                dataset = tf.data.Dataset.from_tensor_slices((images_np, labels_np))
                
                # Configure dataset to use CPU
                options = tf.data.Options()
                options.experimental_optimization.apply_default_optimizations = False
                dataset = dataset.with_options(options)
                
                if shuffle_data:
                    # Shuffle on CPU - use smaller buffer if needed
                    buffer_size = min(10000, len(ds_obj))
                    # Use deterministic=False to avoid GPU issues
                    dataset = dataset.shuffle(buffer_size=buffer_size, seed=42, reshuffle_each_iteration=True)
                
                return dataset
            
            # Create datasets - all operations will be on CPU due to numpy arrays
            # Wrap in CPU device context to be safe
            with tf.device('/CPU:0'):
                train_ds = create_dataset(train_ds_obj, shuffle_data=True)
                train_ds = train_ds.batch(train_bs, drop_remainder=drop_last)
                
                test_ds = create_dataset(test_ds_obj, shuffle_data=False)
                test_ds = test_ds.batch(test_bs)
            
            # Prefetch for better performance
            train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
            test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
            
            return train_ds, test_ds, train_ds_obj, test_ds_obj

        elif dataset == 'cifar100':
            raise NotImplementedError("CIFAR-100 not implemented yet")

    return None, None, None, None


if __name__ == '__main__':
    load_cifar10_data(datadir='./data/')
