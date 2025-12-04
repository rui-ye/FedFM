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

def matching_l2_regression(features, labels, centroids):
    features = paddle.nn.functional.normalize(features)
    centroids = paddle.nn.functional.normalize(centroids)
    labels = get_anchor_labels(labels, centroids.shape[0])

    centroids_new=centroids.unsqueeze(0).repeat(features.shape[0],1,1)
    labels_new=labels.view(features.shape[0],1,1).repeat(1,1,features.shape[1])
    centroids_new=paddle.gather(centroids_new,1,labels_new).squeeze()
    loss=paddle.nn.functional.mse_loss(centroids_new,features)
    return loss

def matching_cross_entropy_regression(features, labels, centroids, tao, only_small=False, dominant_class=None):
    features = paddle.nn.functional.normalize(features)
    centroids = paddle.nn.functional.normalize(centroids)
    labels = get_anchor_labels(labels, centroids.shape[0])

    if only_small:      # yr for xmk: not used, ignore
        small_filter = paddle.BoolTensor(([True] * len(labels))).to(features.device)
        for dominant_class_id in dominant_class:
            small_filter = small_filter & (labels!=dominant_class_id)
        features = features[small_filter]                     # only the small classes are regularized
        labels = labels[small_filter]

    similarity_matrix = paddle.mm(features,(centroids.T))/tao
    loss = paddle.nn.functional.cross_entropy(similarity_matrix, labels)
    return loss

def get_anchor_labels(labels, num_anchor):
    # assign each ground truth label a new label, which corresponds to one anchor
    partition_labels = labels//(100.0/num_anchor)
    return partition_labels


if __name__ == '__main__':
        representation = paddle.load("../test_file/representation.pt",map_location=paddle.device('cpu'))
        labels = paddle.load("../test_file/labels.pt",map_location=paddle.device('cpu'))
        centroids = paddle.load("../test_file/centroids.pt",map_location=paddle.device('cpu'))

        print(representation[:2])
        print(labels[:2])
        print(centroids)


        loss = matching_cross_entropy(representation, labels, centroids, tao=0.1, only_small=False, dominant_class=0)
        print(loss)