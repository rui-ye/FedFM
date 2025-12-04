import tensorflow as tf
import numpy as np


def matching_l2(features, labels, centroids):
    """L2 matching loss"""
    features = tf.nn.l2_normalize(features, axis=1)
    centroids = tf.nn.l2_normalize(centroids, axis=1)
    
    # Gather centroids for each label
    # Convert labels to int32 for indexing
    labels = tf.cast(labels, tf.int32)
    # Use gather_nd or simple indexing
    batch_size = tf.shape(features)[0]
    indices = tf.stack([tf.range(batch_size), labels], axis=1)
    centroids_selected = tf.gather_nd(centroids, indices)
    
    loss = tf.reduce_mean(tf.square(centroids_selected - features))
    return loss


def matching_cross_entropy(features, labels, centroids, tao, only_small=False, dominant_class=None):
    """Cross-entropy matching loss"""
    features = tf.nn.l2_normalize(features, axis=1)
    centroids = tf.nn.l2_normalize(centroids, axis=1)
    
    if only_small:
        small_filter = tf.ones(tf.shape(labels)[0], dtype=tf.bool)
        for dominant_class_id in dominant_class:
            small_filter = tf.logical_and(small_filter, tf.not_equal(labels, dominant_class_id))
        features = tf.boolean_mask(features, small_filter)
        labels = tf.boolean_mask(labels, small_filter)
    
    # Compute similarity matrix: features @ centroids^T / tao
    similarity_matrix = tf.matmul(features, centroids, transpose_b=True) / tao
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=similarity_matrix))
    return loss


def matching_l2_regression(features, labels, centroids):
    """L2 matching loss for regression"""
    features = tf.nn.l2_normalize(features, axis=1)
    centroids = tf.nn.l2_normalize(centroids, axis=1)
    labels = get_anchor_labels(labels, tf.shape(centroids)[0])
    
    labels_expanded = tf.expand_dims(labels, axis=1)
    labels_expanded = tf.expand_dims(labels_expanded, axis=2)
    labels_expanded = tf.tile(labels_expanded, [1, 1, tf.shape(features)[1]])
    
    centroids_expanded = tf.expand_dims(centroids, axis=0)
    centroids_expanded = tf.tile(centroids_expanded, [tf.shape(features)[0], 1, 1])
    
    centroids_selected = tf.gather(centroids_expanded, labels, axis=1, batch_dims=1)
    centroids_selected = tf.squeeze(centroids_selected, axis=1)
    
    loss = tf.reduce_mean(tf.square(centroids_selected - features))
    return loss


def matching_cross_entropy_regression(features, labels, centroids, tao, only_small=False, dominant_class=None):
    """Cross-entropy matching loss for regression"""
    features = tf.nn.l2_normalize(features, axis=1)
    centroids = tf.nn.l2_normalize(centroids, axis=1)
    labels = get_anchor_labels(labels, tf.shape(centroids)[0])
    
    if only_small:
        small_filter = tf.ones(tf.shape(labels)[0], dtype=tf.bool)
        for dominant_class_id in dominant_class:
            small_filter = tf.logical_and(small_filter, tf.not_equal(labels, dominant_class_id))
        features = tf.boolean_mask(features, small_filter)
        labels = tf.boolean_mask(labels, small_filter)
    
    similarity_matrix = tf.matmul(features, centroids, transpose_b=True) / tao
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=similarity_matrix))
    return loss


def get_anchor_labels(labels, num_anchor):
    """Assign each ground truth label a new label, which corresponds to one anchor"""
    partition_labels = tf.cast(labels // (100.0 / num_anchor), tf.int32)
    return partition_labels
