from math import log2
from numpy import partition
import paddle
import paddle.nn.functional as F
import sys

def matching_l2(features, labels, centroids):
    features = paddle.nn.functional.normalize(features)
    centroids = paddle.nn.functional.normalize(centroids)

    centroids_new=centroids.unsqueeze(0).repeat(features.shape[0],1,1)
    labels_new=labels.view(features.shape[0],1,1).repeat(1,1,features.shape[1])
    centroids_new=paddle.gather(centroids_new,1,labels_new).squeeze()
    loss=paddle.nn.functional.mse_loss(centroids_new,features)
    return loss

def matching_cross_entropy(features, labels, centroids, tao, only_small=False, dominant_class=None):
    features = paddle.nn.functional.normalize(features)
    centroids = paddle.nn.functional.normalize(centroids)

    if only_small:
        small_filter = paddle.BoolTensor(([True] * len(labels))).to(features.device)
        for dominant_class_id in dominant_class:
            small_filter = small_filter & (labels!=dominant_class_id)
        features = features[small_filter]                     # only the small classes are regularized
        labels = labels[small_filter]

    similarity_matrix = paddle.mm(features,(centroids.T))/tao
    loss = paddle.nn.functional.cross_entropy(similarity_matrix, labels)
    return loss

