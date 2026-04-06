from __future__ import print_function

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import keras
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

batch_size = 512
num_classes = 10
epochs = 20

# load MNIST
(x_train_full, y_train_full), (x_test, y_test) = mnist.load_data()

# validation slice from train
n_val = 6000
x_val = x_train_full[:n_val].reshape(n_val, 784).astype('float32') / 255.0
y_val = y_train_full[:n_val]
x_train = x_train_full[n_val:].reshape(-1, 784).astype('float32') / 255.0
y_train = y_train_full[n_val:]

x_test = x_test.reshape(10000, 784).astype('float32') / 255.0
print(x_train.shape[0], 'train samples (after val split)')
print(x_val.shape[0], 'validation samples')
print(x_test.shape[0], 'test samples')

# one-hot labels
y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_val_cat = keras.utils.to_categorical(y_val, num_classes)
y_test_cat = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train_cat,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_val, y_val_cat))

score = model.evaluate(x_test, y_test_cat, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# plot train/val curves
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(history.history['accuracy'], label='train')
ax[0].plot(history.history['val_accuracy'], label='validation')
ax[0].set_title('Model accuracy')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Accuracy')
ax[0].legend(loc='lower right')
ax[0].grid(True, alpha=0.3)

ax[1].plot(history.history['loss'], label='train')
ax[1].plot(history.history['val_loss'], label='validation')
ax[1].set_title('Model loss')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Loss')
ax[1].legend(loc='upper right')
ax[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'mnist_training_history.png', dpi=150)
plt.close()

# test confusion matrix
y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
cm = tf.math.confusion_matrix(y_test, y_pred, num_classes=num_classes).numpy()

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(
    xticks=np.arange(num_classes),
    yticks=np.arange(num_classes),
    xlabel='Predicted label',
    ylabel='True label',
    title='MNIST test confusion matrix',
)
plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha='center', va='center',
                color='white' if cm[i, j] > thresh else 'black')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'mnist_confusion_matrix.png', dpi=150)
plt.close()

print('Saved figures to "{}"'.format(OUTPUT_DIR))
