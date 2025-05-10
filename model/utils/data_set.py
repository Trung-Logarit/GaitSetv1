import tensorflow as tf
import numpy as np
import os.path as osp
import os
import xarray as xr

class DataSet(tf.data.Dataset):
    def __init__(self, seq_dir, label, seq_type, view, cache, resolution):
        self.seq_dir = seq_dir
        self.view = view
        self.seq_type = seq_type
        self.label = label
        self.cache = cache
        self.resolution = int(resolution)
        self.cut_padding = int(float(resolution) / 64 * 10)
        self.data_size = len(self.label)
        self.data = [None] * self.data_size
        self.frame_set = [None] * self.data_size

        self.label_set = set(self.label)
        self.seq_type_set = set(self.seq_type)
        self.view_set = set(self.view)

        # Tạo index_dict như trong PyTorch
        _ = np.zeros((len(self.label_set),
                      len(self.seq_type_set),
                      len(self.view_set))).astype('int')
        _ -= 1
        self.index_dict = xr.DataArray(
            _,
            coords={'label': sorted(list(self.label_set)),
                    'seq_type': sorted(list(self.seq_type_set)),
                    'view': sorted(list(self.view_set))},
            dims=['label', 'seq_type', 'view'])

        for i in range(self.data_size):
            _label = self.label[i]
            _seq_type = self.seq_type[i]
            _view = self.view[i]
            self.index_dict.loc[_label, _seq_type, _view] = i

    def load_all_data(self):
        for i in range(self.data_size):
            self.load_data(i)

    def load_data(self, index):
        return self.__getitem__(index)

    def __loader__(self, path):
        return self.img2xarray(path)[:, :, self.cut_padding:-self.cut_padding].astype('float32') / 255.0

    def __getitem__(self, index):
        # Sampling pose sequence
        if not self.cache or self.data[index] is None:
            data = [self.__loader__(_path) for _path in self.seq_dir[index]]
            frame_set = [set(feature.coords['frame'].values.tolist()) for feature in data]
            frame_set = list(set.intersection(*frame_set))
            if self.cache:
                self.data[index] = data
                self.frame_set[index] = frame_set
        else:
            data = self.data[index]
            frame_set = self.frame_set[index]

        return data, frame_set, self.view[index], self.seq_type[index], self.label[index]

    def img2xarray(self, file_path):
        imgs = sorted(list(os.listdir(file_path)))
        frame_list = [
            np.reshape(tf.image.decode_image(tf.io.read_file(osp.join(file_path, img)), channels=1).numpy(),
                       [self.resolution, self.resolution])
            for img in imgs if osp.isfile(osp.join(file_path, img))
        ]
        num_list = list(range(len(frame_list)))
        data_dict = xr.DataArray(
            frame_list,
            coords={'frame': num_list},
            dims=['frame', 'img_y', 'img_x'],
        )
        return data_dict

    def __len__(self):
        return len(self.label)

print("✅ DataSet đã sẵn sàng!")