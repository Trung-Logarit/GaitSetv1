import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from .network.gaitset_keras import SetNet
from .network.triplet_loss_keras import TripletLoss
from .utils.sampler_keras import TripletSampler


class Model:
    def __init__(self,
                 hidden_dim,
                 lr,
                 hard_or_full_trip,
                 margin,
                 num_workers,
                 batch_size,
                 restore_iter,
                 total_iter,
                 save_name,
                 train_pid_num,
                 frame_num,
                 model_name,
                 train_source,
                 test_source,
                 img_size=64):

        self.save_name = save_name
        self.train_pid_num = train_pid_num
        self.train_source = train_source
        self.test_source = test_source

        self.hidden_dim = hidden_dim
        self.lr = lr
        self.hard_or_full_trip = hard_or_full_trip
        self.margin = margin
        self.frame_num = frame_num
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.model_name = model_name
        self.P, self.M = batch_size

        self.restore_iter = restore_iter
        self.total_iter = total_iter

        self.img_size = img_size

        # Khởi tạo encoder và loss
        self.encoder = SetNet(hidden_dim=self.hidden_dim)
        self.triplet_loss = TripletLoss(self.P * self.M, self.hard_or_full_trip, self.margin)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

        self.hard_loss_metric = []
        self.full_loss_metric = []
        self.full_loss_num = []
        self.dist_list = []
        self.mean_dist = 0.01

        self.sample_type = 'all'

    def collate_fn(self, batch):
        batch_size = len(batch)
        feature_num = len(batch[0][0])
        seqs = [batch[i][0] for i in range(batch_size)]
        frame_sets = [batch[i][1] for i in range(batch_size)]
        view = [batch[i][2] for i in range(batch_size)]
        seq_type = [batch[i][3] for i in range(batch_size)]
        label = [batch[i][4] for i in range(batch_size)]
        batch = [seqs, view, seq_type, label, None]

        def select_frame(index):
            sample = seqs[index]
            frame_set = frame_sets[index]
            if self.sample_type == 'random':
                frame_id_list = np.random.choice(list(frame_set), self.frame_num, replace=True)
                _ = [feature.loc[frame_id_list].values for feature in sample]
            else:
                _ = [feature.values for feature in sample]
            return _

        seqs = list(map(select_frame, range(len(seqs))))

        if self.sample_type == 'random':
            seqs = [np.asarray([seqs[i][j] for i in range(batch_size)]) for j in range(feature_num)]
        else:
            seqs = [np.concatenate([seqs[i][j] for i in range(batch_size)], axis=0) for j in range(feature_num)]

        batch[0] = seqs
        return batch

    def fit(self):
        if self.restore_iter != 0:
            self.load(self.restore_iter)

        print("🔄 Bắt đầu huấn luyện...")
        sampler = TripletSampler(self.train_source, self.batch_size)
        train_data = tf.data.Dataset.from_generator(
            sampler,
            output_signature=tf.TensorSpec(shape=(self.P * self.M,), dtype=tf.int32)
        )

        for step, sample_indices in enumerate(train_data):
            self.restore_iter += 1
            batch_data = [self.train_source[i] for i in sample_indices.numpy()]
            batch_features, batch_labels = self.collate_fn(batch_data)

            with tf.GradientTape() as tape:
                features, _ = self.encoder(batch_features)
                full_loss_metric, hard_loss_metric, mean_dist, full_loss_num = self.triplet_loss(features, batch_labels)
                loss = tf.reduce_mean(hard_loss_metric if self.hard_or_full_trip == 'hard' else full_loss_metric)

            grads = tape.gradient(loss, self.encoder.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.encoder.trainable_variables))

            self.hard_loss_metric.append(tf.reduce_mean(hard_loss_metric).numpy())
            self.full_loss_metric.append(tf.reduce_mean(full_loss_metric).numpy())
            self.full_loss_num.append(tf.reduce_mean(full_loss_num).numpy())
            self.dist_list.append(tf.reduce_mean(mean_dist).numpy())

            if self.restore_iter % 100 == 0:
                print(f"Step {self.restore_iter}, Loss: {loss.numpy():.4f}, Hard Loss: {np.mean(self.hard_loss_metric):.4f}, Full Loss: {np.mean(self.full_loss_metric):.4f}")
                self.hard_loss_metric = []
                self.full_loss_metric = []
                self.full_loss_num = []
                self.dist_list = []

            if self.restore_iter == self.total_iter:
                print("✅ Huấn luyện hoàn tất!")
                break

    def save(self):
        os.makedirs(os.path.join('checkpoint', self.model_name), exist_ok=True)
        self.encoder.save_weights(os.path.join('checkpoint', self.model_name, f"{self.save_name}-encoder.h5"))
        print(f"✅ Đã lưu mô hình tại checkpoint/{self.model_name}/")

    def load(self, restore_iter):
        self.encoder.load_weights(os.path.join('checkpoint', self.model_name, f"{self.save_name}-encoder.h5"))
        print(f"✅ Đã tải mô hình từ checkpoint/{self.model_name}/")


print("✅ Model đã sẵn sàng!")