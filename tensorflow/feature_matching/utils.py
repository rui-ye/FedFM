import tensorflow as tf
import numpy as np


def get_client_centroids_info(model, dataloaders, model_name, dataset_name, party_list_this_round, num_anchor=0):
    """Return the centroids and num_per_class for each client"""
    model.trainable = False
    
    local_centroids = []
    local_distributions = []

    if model_name.startswith('resnet50'):
        feature_d = 2048
    elif model_name.startswith('resnet18'):
        feature_d = 512

    if dataset_name in ['cifar10', 'cinic10']:
        num_classes = 10
    elif dataset_name == 'cifar100':
        num_classes = 100
    elif dataset_name == 'tinyimagenet':
        num_classes = 200
    elif dataset_name == 'ham10000':
        num_classes = 7
    elif dataset_name == 'wiki':
        num_classes = num_anchor

    for net_id in party_list_this_round:
        dataset = dataloaders[net_id]
        dataset_size = len(list(dataset))
        client_rep = []
        client_label = []

        for images, labels in dataset:
            # Get representation from model
            representation, _, _ = model(images, training=False)
            client_rep.append(representation.numpy())
            client_label.append(labels.numpy())

        client_rep = np.concatenate(client_rep, axis=0)
        client_label = np.concatenate(client_label, axis=0)
        
        # Ensure we have the right size
        if len(client_rep) > dataset_size:
            client_rep = client_rep[:dataset_size]
            client_label = client_label[:dataset_size]

        client_centroids, client_distribution = cal_center(
            rep=tf.constant(client_rep), 
            label=tf.constant(client_label), 
            num_classes=num_classes
        )

        local_centroids.append(client_centroids.numpy())
        local_distributions.append(client_distribution.numpy())

    model.trainable = True

    return local_centroids, local_distributions


def cal_center(rep, label, num_classes):
    """Calculation for the 'get_client_centroids_info' function"""
    center_of_class = tf.zeros((num_classes, rep.shape[1]))
    distribution_of_class = tf.zeros(num_classes)
    
    for class_id in range(num_classes):
        mask = tf.equal(label, class_id)
        count = tf.reduce_sum(tf.cast(mask, tf.float32))
        if count != 0:
            # Update center: indices should be [[class_id]] for 2D tensor
            mean_rep = tf.reduce_mean(tf.boolean_mask(rep, mask), axis=0)
            center_of_class = tf.tensor_scatter_nd_update(
                center_of_class,
                [[class_id]],  # indices shape: [1, 1] for 2D tensor
                [mean_rep]  # updates shape: [1, rep.shape[1]]
            )
            # Update distribution: indices should be [[class_id]] for 1D tensor
            distribution_of_class = tf.tensor_scatter_nd_update(
                distribution_of_class,
                [[class_id]],  # indices shape: [1, 1] for 1D tensor
                [count]  # updates shape: [1]
            )

    return center_of_class, distribution_of_class


def get_global_centroids(local_centroids, local_distributions, pre_global_centroids, momentum=0.0, equally_average=0):
    """Calculate global centroids using local_centroids based on local_distributions"""
    local_centroids = [tf.constant(c) for c in local_centroids]
    local_distributions = [tf.constant(d) for d in local_distributions]
    pre_global_centroids = tf.constant(pre_global_centroids)
    
    global_centroids = tf.zeros_like(local_centroids[0])

    if equally_average:
        # If a client has no data sample of category x, then assign the corresponding local anchor 
        # with last round's global anchor
        for client_id in range(len(local_centroids)):
            for class_id in range(global_centroids.shape[0]):
                if local_distributions[client_id][class_id] < 1:
                    local_centroids[client_id] = tf.tensor_scatter_nd_update(
                        local_centroids[client_id],
                        [[class_id]],
                        [pre_global_centroids[class_id]]
                    )
        for client_id in range(len(local_centroids)):
            global_centroids += local_centroids[client_id] / len(local_centroids)
    else:
        for class_id in range(global_centroids.shape[0]):
            total_num = 0.0
            for client_id in range(len(local_centroids)):
                total_num += local_distributions[client_id][class_id]
            
            class_centroid = tf.zeros(global_centroids.shape[1])
            for client_id in range(len(local_centroids)):
                if total_num > 0:
                    weight = local_distributions[client_id][class_id] / total_num
                    class_centroid += local_centroids[client_id][class_id] * weight
            
            global_centroids = tf.tensor_scatter_nd_update(
                global_centroids,
                [[class_id]],
                [class_centroid]
            )
    
    if tf.reduce_sum(global_centroids) == 0:
        print('First round requires no pre_global_centroids.')
    else:
        global_centroids = momentum * pre_global_centroids + (1 - momentum) * global_centroids

    return global_centroids.numpy()


def personalized_get_client_centroids_info(nets_this_round, dataloaders, model_name, dataset_name, party_list_this_round, num_anchor=0):
    """Return the centroids and num_per_class for each client (personalized version)"""
    local_centroids = []
    local_distributions = []

    if model_name == 'resnet50' or model_name == 'resnet50_7':
        feature_d = 2048
    elif model_name == 'resnet18' or model_name == 'resnet18_7':
        feature_d = 512

    if dataset_name == 'cifar10':
        num_classes = 10
    elif dataset_name == 'cifar100':
        num_classes = 100
    elif dataset_name == 'tinyimagenet':
        num_classes = 200
    elif dataset_name == 'ham10000':
        num_classes = 7
    elif dataset_name == 'wiki':
        num_classes = num_anchor

    for net_id in party_list_this_round:
        model = nets_this_round[net_id]
        model.trainable = False

        dataset = dataloaders[net_id]
        client_rep = []
        client_label = []

        for images, labels in dataset:
            representation, _, _ = model(images, training=False)
            client_rep.append(representation.numpy())
            client_label.append(labels.numpy())

        client_rep = np.concatenate(client_rep, axis=0)
        client_label = np.concatenate(client_label, axis=0)

        client_centroids, client_distribution = cal_center(
            rep=tf.constant(client_rep), 
            label=tf.constant(client_label), 
            num_classes=num_classes
        )

        local_centroids.append(client_centroids.numpy())
        local_distributions.append(client_distribution.numpy())
        
        model.trainable = True

    return local_centroids, local_distributions
