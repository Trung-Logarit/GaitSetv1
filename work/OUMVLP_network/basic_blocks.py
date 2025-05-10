import tensorflow as tf
from tensorflow.keras import layers, Model

class BasicConv2D(layers.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv2D, self).__init__()
        self.conv = layers.Conv2D(out_channels, kernel_size, use_bias=False, **kwargs)
        self.activation = layers.LeakyReLU()

    def call(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class SetBlock(layers.Layer):
    def __init__(self, forward_block, pooling=False):
        super(SetBlock, self).__init__()
        self.forward_block = forward_block
        self.pooling = pooling
        if pooling:
            self.pool2d = layers.MaxPooling2D(pool_size=(2, 2))

    def call(self, x):
        # Chuyển từ (n, s, c, h, w) -> (n * s, h, w, c)
        n, s, c, h, w = x.shape
        x = tf.reshape(x, (-1, h, w, c))
        x = self.forward_block(x)
        if self.pooling:
            x = self.pool2d(x)
        # Trả về (n, s, c, h, w)
        c, h, w = x.shape[-3:]
        x = tf.reshape(x, (n, s, c, h, w))
        return x


class HPM(layers.Layer):
    def __init__(self, in_dim, out_dim, bin_level_num=5):
        super(HPM, self).__init__()
        self.bin_num = [2 ** i for i in range(bin_level_num)]
        self.fc_bin = self.add_weight(
            shape=(sum(self.bin_num), in_dim, out_dim),
            initializer="glorot_uniform",
            trainable=True,
            name="fc_bin"
        )

    def call(self, x):
        features = []
        n, c, h, w = x.shape
        for num_bin in self.bin_num:
            z = tf.reshape(x, (n, c, num_bin, -1))
            z = tf.reduce_mean(z, axis=3) + tf.reduce_max(z, axis=3)
            features.append(z)
        features = tf.concat(features, axis=2)
        features = tf.einsum('ncl,lcd->ncd', features, self.fc_bin)
        return features


print("✅ BasicConv2D, SetBlock và HPM đã sẵn sàng!")