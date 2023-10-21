from math import log2
from numpy import partition
# import torch
import mindspore
# import torch.nn.functional as F
import sys

def matching_l2(features, labels, centroids):
    features = mindspore.ops.L2Normalize(features)
    centroids = mindspore.ops.L2Normalize(centroids)

    centroids_new=centroids.unsqueeze(0).repeat(features.shape[0],1,1)
    labels_new=labels.view(features.shape[0],1,1).repeat(1,1,features.shape[1])
    centroids_new=mindspore.ops.gather_elements(centroids_new,1,labels_new).squeeze()
    loss=mindspore.nn.MSELoss(centroids_new,features)
    return loss

def matching_cross_entropy(features, labels, centroids, tao, only_small=False, dominant_class=None):
    features = mindspore.ops.L2Normalize(axis=1,epsilon=1e-12)(features)
    centroids = mindspore.ops.L2Normalize(axis=1,epsilon=1e-12)(centroids)
    # tao_tensor = mindspore.Tensor(tao)
    if only_small:
        small_filter = mindspore.Tensor.all(([True] * len(labels))) # .to(features.device)
        for dominant_class_id in dominant_class:
            small_filter = small_filter & (labels!=dominant_class_id)
        features = features[small_filter]                     # only the small classes are regularized
        labels = labels[small_filter]

    matmul = mindspore.ops.MatMul(False,True)
    similarity_matrix = matmul(features,centroids) #original (centroids.T)
    similarity_matrix = similarity_matrix/0.1 # tau equals 0.1
    loss = mindspore.ops.cross_entropy(similarity_matrix, labels)
    return loss

