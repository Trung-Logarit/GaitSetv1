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

print("✅ BasicConv2D và SetBlock đã sẵn sàng!")