import tensorflow as tf
from tensorflow.keras import layers, Model
from .basic_blocks_keras import SetBlock, BasicConv2D, HPM

class SetNet(Model):
    def __init__(self, hidden_dim):
        super(SetNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_frame = None
        
        _in_channels = 1
        _channels = [64, 128, 256]
        self.set_layer1 = SetBlock(BasicConv2D(_in_channels, _channels[0], 5, padding="same"))
        self.set_layer2 = SetBlock(BasicConv2D(_channels[0], _channels[0], 3, padding="same"), pooling=True)
        self.set_layer3 = SetBlock(BasicConv2D(_channels[0], _channels[1], 3, padding="same"))
        self.set_layer4 = SetBlock(BasicConv2D(_channels[1], _channels[1], 3, padding="same"), pooling=True)
        self.set_layer5 = SetBlock(BasicConv2D(_channels[1], _channels[2], 3, padding="same"))
        self.set_layer6 = SetBlock(BasicConv2D(_channels[2], _channels[2], 3, padding="same"))
        
        self.gl_layer1 = BasicConv2D(_channels[0], _channels[1], 3, padding="same")
        self.gl_layer2 = BasicConv2D(_channels[1], _channels[1], 3, padding="same")
        self.gl_layer3 = BasicConv2D(_channels[1], _channels[2], 3, padding="same")
        self.gl_layer4 = BasicConv2D(_channels[2], _channels[2], 3, padding="same")
        self.gl_pooling = layers.MaxPooling2D(pool_size=(2, 2))
        
        self.gl_hpm = HPM(_channels[-1], hidden_dim)
        self.x_hpm = HPM(_channels[-1], hidden_dim)

    def frame_max(self, x):
        return tf.reduce_max(x, axis=1)

    def call(self, silho, batch_frame=None):
        silho = silho / 255.0  # Chuẩn hóa
        n, s, h, w, c = silho.shape
        x = tf.reshape(silho, (n * s, h, w, c))

        x = self.set_layer1(x)
        x = self.set_layer2(x)
        gl = self.gl_layer1(self.frame_max(x))
        gl = self.gl_layer2(gl)
        gl = self.gl_pooling(gl)
        
        x = self.set_layer3(x)
        x = self.set_layer4(x)
        gl = self.gl_layer3(gl + self.frame_max(x))
        gl = self.gl_layer4(gl)

        x = self.set_layer5(x)
        x = self.set_layer6(x)
        x = self.frame_max(x)
        gl = gl + x
        
        gl_f = self.gl_hpm(gl)
        x_f = self.x_hpm(x)

        # Kết hợp đặc trưng global và local
        return tf.concat([gl_f, x_f], axis=1), None

print("✅ SetNet đã sẵn sàng!")
