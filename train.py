import os
import tensorflow as tf
from model.network.gaitset_keras import GaitSetEncoder
from config import conf
from pretreatment_keras import preprocess_dataset

# Đường dẫn dữ liệu
DATA_PATH = conf["data"]["dataset_path"]
CHECKPOINT_PATH = conf["WORK_PATH"] + "/checkpoint/"
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

# Khởi tạo mô hình
model = GaitSetEncoder(hidden_dim=conf["model"]["hidden_dim"])
model.build((None, 64, 64, 1))
model.summary()

# Chuẩn bị dữ liệu
print("🔄 Đang chuẩn bị dữ liệu...")
preprocess_dataset(DATA_PATH, DATA_PATH + "_processed")

train_dataset = tf.data.Dataset.list_files(DATA_PATH + "_processed/*/*.png")
train_dataset = train_dataset.map(lambda x: (tf.image.decode_png(tf.io.read_file(x), channels=1), 0))
train_dataset = train_dataset.batch(conf["model"]["batch_size"][0]).prefetch(tf.data.AUTOTUNE)

# Cấu hình optimizer và loss
optimizer = tf.keras.optimizers.Adam(learning_rate=conf["model"]["lr"])
loss_fn = tf.keras.losses.CategoricalCrossentropy()

# Cấu hình callback
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    CHECKPOINT_PATH + "gaitset_encoder.h5", save_best_only=True)
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

# Huấn luyện mô hình
print("🔄 Đang huấn luyện mô hình...")
model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])
model.fit(train_dataset, epochs=conf["model"]["total_iter"] // 1000,
          callbacks=[checkpoint_cb, early_stopping_cb])

print("✅ Huấn luyện hoàn tất!")