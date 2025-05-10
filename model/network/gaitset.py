import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from .basic_blocks_keras import SetBlock, BasicConv2D

class SetNet(Model):
    def __init__(self, hidden_dim):
        super(SetNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_frame = None

        _set_in_channels = 1
        _set_channels = [32, 64, 128]
        self.set_layer1 = SetBlock(BasicConv2D(_set_in_channels, _set_channels[0], 5, padding="same"))
        self.set_layer2 = SetBlock(BasicConv2D(_set_channels[0], _set_channels[0], 3, padding="same"), pooling=True)
        self.set_layer3 = SetBlock(BasicConv2D(_set_channels[0], _set_channels[1], 3, padding="same"))
        self.set_layer4 = SetBlock(BasicConv2D(_set_channels[1], _set_channels[1], 3, padding="same"), pooling=True)
        self.set_layer5 = SetBlock(BasicConv2D(_set_channels[1], _set_channels[2], 3, padding="same"))
        self.set_layer6 = SetBlock(BasicConv2D(_set_channels[2], _set_channels[2], 3, padding="same"))

        _gl_in_channels = 32
        _gl_channels = [64, 128]
        self.gl_layer1 = BasicConv2D(_gl_in_channels, _gl_channels[0], 3, padding="same")
        self.gl_layer2 = BasicConv2D(_gl_channels[0], _gl_channels[0], 3, padding="same")
        self.gl_layer3 = BasicConv2D(_gl_channels[0], _gl_channels[1], 3, padding="same")
        self.gl_layer4 = BasicConv2D(_gl_channels[1], _gl_channels[1], 3, padding="same")
        self.gl_pooling = layers.MaxPooling2D(pool_size=(2, 2))

        self.bin_num = [1, 2, 4, 8, 16]
        self.fc_bin = self.add_weight(
            shape=(sum(self.bin_num) * 2, 128, hidden_dim),
            initializer="glorot_uniform",
            trainable=True,
            name="fc_bin"
        )

    def frame_max(self, x):
        if self.batch_frame is None:
            return tf.reduce_max(x, axis=1), None
        else:
            max_list = []
            for i in range(len(self.batch_frame) - 1):
                max_list.append(tf.reduce_max(x[:, self.batch_frame[i]:self.batch_frame[i + 1], :, :, :], axis=1))
            return tf.concat(max_list, axis=0), None

    def call(self, silho, batch_frame=None):
        # Xử lý batch frame
        if batch_frame is not None:
            batch_frame = batch_frame[0].numpy().tolist()
            batch_frame = [f for f in batch_frame if f > 0]
            self.batch_frame = [0] + np.cumsum(batch_frame).tolist()

        # Bắt đầu forward pass
        x = tf.expand_dims(silho, axis=2)  # (n, s, 1, h, w)
        x = self.set_layer1(x)
        x = self.set_layer2(x)
        gl = self.gl_layer1(self.frame_max(x)[0])
        gl = self.gl_layer2(gl)
        gl = self.gl_pooling(gl)

        x = self.set_layer3(x)
        x = self.set_layer4(x)
        gl = self.gl_layer3(gl + self.frame_max(x)[0])
        gl = self.gl_layer4(gl)

        x = self.set_layer5(x)
        x = self.set_layer6(x)
        x = self.frame_max(x)[0]
        gl = gl + x

        # Tạo feature vector
        features = []
        n, c, h, w = gl.shape
        for num_bin in self.bin_num:
            z = tf.reshape(x, (n, c, num_bin, -1))
            z = tf.reduce_mean(z, axis=3) + tf.reduce_max(z, axis=3)
            features.append(z)

            z = tf.reshape(gl, (n, c, num_bin, -1))
            z = tf.reduce_mean(z, axis=3) + tf.reduce_max(z, axis=3)
            features.append(z)

        features = tf.concat(features, axis=2)
        features = tf.einsum('ncl,lcd->ncd', features, self.fc_bin)
        return features, None

print("✅ SetNet đã sẵn sàng!")