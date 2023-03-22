import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import tf2onnx

from mine import p1

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# print(p3(x_train))
# print(p4(y_train))
# print(p3(x_test))
# print(p4(y_test))

batch_size = 64

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
# print(p1(train_dataset))
# print(pNestedSeq2(train_dataset, 2))
# print(p1(train_dataset.cardinality()))

val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
# print(p1(val_dataset))
# print(pNestedSeq2(val_dataset, 2))
# print(p1(val_dataset.cardinality()))

inputs = keras.Input(shape=(28, 28))
x1 = layers.Rescaling(1.0 / 255)(inputs)
x2 = layers.Flatten()(x1)
x3 = layers.Dense(128, activation="relu")(x2)
x4 = layers.Dense(128, activation="relu")(x3)
outputs = layers.Dense(10, activation="softmax")(x4)
model = keras.Model(inputs, outputs)
print(p1(model.input))
print(p1(model.output))
print("\n")
model.summary()

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    run_eagerly=False
)

class InterruptingCallback(tf.keras.callbacks.Callback):
  def on_epoch_begin(self, epoch, logs=None):
    if (epoch + 1) % 2 == 0:
      raise RuntimeError('Interrupting!')

MODELCHECKPOINT_PATH = 'mnist/ModelCheckpoint/best'
BACKUPANDRESTORE_PATH = "mnist/BackupAndRestore"
TENSORBOARD_PATH = "mnist/TensorBoard"
CSVLOGGER_PATH = "mnist/CSVLogger.csv"
BEST_MODEL_PATH = "mnist/best"
BEST_ONNX_MODEL_PATH = "mnist/best.onnx"

try:
  df = pd.read_csv(CSVLOGGER_PATH)
  prev_min = df["val_loss"].min()
except FileNotFoundError:
  prev_min = np.inf
  
callbacks = [
    keras.callbacks.BackupAndRestore(
      backup_dir=BACKUPANDRESTORE_PATH,
      delete_checkpoint=False
      ),

    keras.callbacks.ModelCheckpoint(
        filepath=MODELCHECKPOINT_PATH,
        verbose=1,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        mode='auto',
        save_freq='epoch',
        initial_value_threshold=prev_min,
        ),

    keras.callbacks.EarlyStopping(
      monitor='val_loss',
      min_delta=0,
      patience=3,
      verbose=1,
      mode='auto',
      restore_best_weights=False,
      start_from_epoch=0
      ),

    keras.callbacks.TensorBoard(log_dir=TENSORBOARD_PATH),

    keras.callbacks.TerminateOnNaN(),

    keras.callbacks.CSVLogger(CSVLOGGER_PATH, append=True),

    InterruptingCallback()
]

epochs = 100

try:
  print("\n", "model.fit(...)")
  history = model.fit(train_dataset,
                      epochs=epochs,
                      validation_data=val_dataset,
                      callbacks=callbacks)
except RuntimeError:
  print("\n", "!!!!!!!!!!!!!!!!!!!!RuntimeError!!!!!!!!!!!!!!!!!!!!")
  callbacks.pop()

  history = model.fit(train_dataset,
                      epochs=epochs,
                      validation_data=val_dataset,
                      callbacks=callbacks)

print(p1(history.history))

print("\n", "model.load_weights(MODELCHECKPOINT_PATH)")
model.load_weights(MODELCHECKPOINT_PATH)

print("\n", "model.evaluate(val_dataset)")
result = model.evaluate(val_dataset)
print(p1(result))

print("\n", "model.save(...)")
model.save(BEST_MODEL_PATH)

print("\n", "tf2onnx.convert.from_keras(...)")
tf2onnx.convert.from_keras(model, output_path=BEST_ONNX_MODEL_PATH)
