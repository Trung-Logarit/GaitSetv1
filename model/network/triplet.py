import tensorflow as tf
from tensorflow.keras import layers, Model

class TripletLoss(layers.Layer):
    def __init__(self, batch_size, hard_or_full, margin):
        super(TripletLoss, self).__init__()
        self.batch_size = batch_size
        self.margin = margin
        self.hard_or_full = hard_or_full

    def call(self, feature, label):
        # feature: [n, m, d], label: [n, m]
        n, m, d = feature.shape

        # Tạo mask cho positive và negative pairs
        label_exp = tf.expand_dims(label, 2)
        hp_mask = tf.equal(label_exp, tf.transpose(label_exp, perm=[0, 2, 1]))  # Positive pairs
        hn_mask = tf.not_equal(label_exp, tf.transpose(label_exp, perm=[0, 2, 1]))  # Negative pairs

        # Tính khoảng cách
        dist = self.batch_dist(feature)  # [n, m, m]
        mean_dist = tf.reduce_mean(dist)
        dist = tf.reshape(dist, (-1,))

        # Hard mining
        hard_hp_dist = tf.reduce_max(tf.reshape(tf.boolean_mask(dist, hp_mask), (n, m, -1)), axis=2)
        hard_hn_dist = tf.reduce_min(tf.reshape(tf.boolean_mask(dist, hn_mask), (n, m, -1)), axis=2)
        hard_loss_metric = tf.nn.relu(self.margin + hard_hp_dist - hard_hn_dist)
        hard_loss_metric_mean = tf.reduce_mean(hard_loss_metric, axis=1)

        # Full mining
        full_hp_dist = tf.reshape(tf.boolean_mask(dist, hp_mask), (n, m, -1, 1))
        full_hn_dist = tf.reshape(tf.boolean_mask(dist, hn_mask), (n, m, 1, -1))
        full_loss_metric = tf.nn.relu(self.margin + full_hp_dist - full_hn_dist)

        full_loss_metric_sum = tf.reduce_sum(full_loss_metric, axis=[1, 2])
        full_loss_num = tf.reduce_sum(tf.cast(full_loss_metric > 0, tf.float32), axis=[1, 2])
        full_loss_metric_mean = full_loss_metric_sum / (full_loss_num + 1e-6)  # Tránh chia cho 0

        return full_loss_metric_mean, hard_loss_metric_mean, mean_dist, full_loss_num

    def batch_dist(self, x):
        # Tính khoảng cách Euclidean giữa các vector
        x2 = tf.reduce_sum(tf.square(x), axis=2, keepdims=True)
        dist = x2 + tf.transpose(x2, perm=[0, 2, 1]) - 2 * tf.matmul(x, x, transpose_b=True)
        dist = tf.sqrt(tf.nn.relu(dist))  # Đảm bảo giá trị không âm
        return dist

print("✅ TripletLoss đã sẵn sàng!")