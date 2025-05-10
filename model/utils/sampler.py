import tensorflow as tf
import random

class TripletSampler(tf.keras.utils.Sequence):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            sample_indices = []
            # Chọn ngẫu nhiên các pid cho batch
            pid_list = random.sample(
                list(self.dataset.label_set),
                self.batch_size[0]
            )
            for pid in pid_list:
                _index = self.dataset.index_dict.loc[pid, :, :].values
                _index = _index[_index >= 0].flatten().tolist()
                # Chọn ngẫu nhiên frame cho mỗi pid
                _index = random.choices(_index, k=self.batch_size[1])
                sample_indices.extend(_index)
            yield sample_indices

    def __len__(self):
        return self.dataset.data_size

print("✅ TripletSampler đã sẵn sàng!")