from math import log2
from numpy import partition
import torch
import torch.nn.functional as F
import sys

def matching_l2(features, labels, centroids):
    features = torch.nn.functional.normalize(features)
    centroids = torch.nn.functional.normalize(centroids)

    centroids_new=centroids.unsqueeze(0).repeat(features.shape[0],1,1)
    labels_new=labels.view(features.shape[0],1,1).repeat(1,1,features.shape[1])
    centroids_new=torch.gather(centroids_new,1,labels_new).squeeze()
    loss=torch.nn.functional.mse_loss(centroids_new,features)
    return loss

def matching_cross_entropy(features, labels, centroids, tao, only_small=False, dominant_class=None):
    features = torch.nn.functional.normalize(features)
    centroids = torch.nn.functional.normalize(centroids)

    if only_small:
        small_filter = torch.BoolTensor(([True] * len(labels))).to(features.device)
        for dominant_class_id in dominant_class:
            small_filter = small_filter & (labels!=dominant_class_id)
        features = features[small_filter]                     # only the small classes are regularized
        labels = labels[small_filter]

    similarity_matrix = torch.mm(features,(centroids.T))/tao
    loss = torch.nn.functional.cross_entropy(similarity_matrix, labels)
    return loss

def matching_l2_regression(features, labels, centroids):
    features = torch.nn.functional.normalize(features)
    centroids = torch.nn.functional.normalize(centroids)
    labels = get_anchor_labels(labels, centroids.shape[0])

    centroids_new=centroids.unsqueeze(0).repeat(features.shape[0],1,1)
    labels_new=labels.view(features.shape[0],1,1).repeat(1,1,features.shape[1])
    centroids_new=torch.gather(centroids_new,1,labels_new).squeeze()
    loss=torch.nn.functional.mse_loss(centroids_new,features)
    return loss

def matching_cross_entropy_regression(features, labels, centroids, tao, only_small=False, dominant_class=None):
    features = torch.nn.functional.normalize(features)
    centroids = torch.nn.functional.normalize(centroids)
    labels = get_anchor_labels(labels, centroids.shape[0])

    if only_small:      # yr for xmk: not used, ignore
        small_filter = torch.BoolTensor(([True] * len(labels))).to(features.device)
        for dominant_class_id in dominant_class:
            small_filter = small_filter & (labels!=dominant_class_id)
        features = features[small_filter]                     # only the small classes are regularized
        labels = labels[small_filter]

    similarity_matrix = torch.mm(features,(centroids.T))/tao
    loss = torch.nn.functional.cross_entropy(similarity_matrix, labels)
    return loss

def get_anchor_labels(labels, num_anchor):
    # assign each ground truth label a new label, which corresponds to one anchor
    partition_labels = labels//(100.0/num_anchor)
    return partition_labels


if __name__ == '__main__':
        representation = torch.load("../test_file/representation.pt",map_location=torch.device('cpu'))
        labels = torch.load("../test_file/labels.pt",map_location=torch.device('cpu'))
        centroids = torch.load("../test_file/centroids.pt",map_location=torch.device('cpu'))

        print(representation[:2])
        print(labels[:2])
        print(centroids)


        loss = matching_cross_entropy(representation, labels, centroids, tao=0.1, only_small=False, dominant_class=0)
        print(loss)