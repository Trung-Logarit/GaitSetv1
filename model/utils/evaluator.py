import tensorflow as tf
import numpy as np


def cuda_dist(x, y):
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.float32)
    dist = tf.reduce_sum(tf.square(x), axis=1, keepdims=True) + \
           tf.reduce_sum(tf.square(y), axis=1, keepdims=True) - 2 * tf.matmul(x, tf.transpose(y))
    dist = tf.sqrt(tf.nn.relu(dist))  # Đảm bảo giá trị không âm
    return dist


def evaluation(data, config):
    dataset = config['dataset'].split('-')[0]
    feature, view, seq_type, label = data
    label = np.array(label)
    view_list = sorted(list(set(view)))
    view_num = len(view_list)

    probe_seq_dict = {'CASIA': [['nm-05', 'nm-06'], ['bg-01', 'bg-02'], ['cl-01', 'cl-02']],
                      'OUMVLP': [['00']]}
    gallery_seq_dict = {'CASIA': [['nm-01', 'nm-02', 'nm-03', 'nm-04']],
                        'OUMVLP': [['01']]}

    num_rank = 5
    acc = np.zeros([len(probe_seq_dict[dataset]), view_num, view_num, num_rank])

    for p, probe_seq in enumerate(probe_seq_dict[dataset]):
        for gallery_seq in gallery_seq_dict[dataset]:
            for v1, probe_view in enumerate(view_list):
                for v2, gallery_view in enumerate(view_list):
                    # Lọc gallery và probe
                    gseq_mask = np.isin(seq_type, gallery_seq) & np.isin(view, [gallery_view])
                    gallery_x = feature[gseq_mask]
                    gallery_y = label[gseq_mask]

                    pseq_mask = np.isin(seq_type, probe_seq) & np.isin(view, [probe_view])
                    probe_x = feature[pseq_mask]
                    probe_y = label[pseq_mask]

                    # Tính khoảng cách và tính rank
                    dist = cuda_dist(probe_x, gallery_x).numpy()
                    idx = np.argsort(dist, axis=1)
                    acc[p, v1, v2, :] = np.round(
                        np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, :num_rank]], axis=1) > 0,
                               axis=0) * 100 / dist.shape[0], 2)

    return acc


print("✅ Evaluator đã sẵn sàng!")
