from __future__ import print_function

import keras
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Rescaling, BatchNormalization
from keras.optimizers import RMSprop, Adam
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

batch_size = 16
num_classes = 3
epochs = 15
img_width = 224
img_height = 224
img_channels = 3
fit = True
train_dir = str(SCRIPT_DIR / 'chest_xray' / 'train')
test_dir = str(SCRIPT_DIR / 'chest_xray' / 'test')

with tf.device('/gpu:0'):

    train_ds, val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        seed=123,
        validation_split=0.2,
        subset='both',
        image_size=(img_height, img_width),
        batch_size=batch_size,
        labels='inferred',
        shuffle=True)

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        seed=None,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        labels='inferred',
        shuffle=True)

    class_names = train_ds.class_names
    print('Class Names: ', class_names)
    num_classes = len(class_names)

    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(2):
        for i in range(6):
            ax = plt.subplot(2, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i].numpy()])
            plt.axis("off")
    plt.savefig(OUTPUT_DIR / 'pneumonia_train_preview.png', dpi=150)
    plt.close()

    model = tf.keras.models.Sequential([
        Rescaling(1.0/255),
        Conv2D(32, (3,3), activation='relu', input_shape=(img_height, img_width, img_channels)),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])

    earlystop_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True)
    save_callback = tf.keras.callbacks.ModelCheckpoint(
        str(OUTPUT_DIR / "pneumonia.keras"), save_freq='epoch', save_best_only=True)
    lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

    if fit:
        history = model.fit(
            train_ds,
            batch_size=batch_size,
            validation_data=val_ds,
            callbacks=[save_callback, earlystop_callback, lr_reduce],
            epochs=epochs)
    else:
        model = tf.keras.models.load_model(str(OUTPUT_DIR / "pneumonia.keras"))

    score = model.evaluate(test_ds, batch_size=batch_size)
    print('Test accuracy:', score[1])

    if fit:
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(OUTPUT_DIR / 'pneumonia_accuracy.png', dpi=150)
        plt.close()

    test_batch = test_ds.take(1)
    plt.figure(figsize=(10, 10))
    for images, labels in test_batch:
        for i in range(6):
            ax = plt.subplot(2, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            prediction = model.predict(tf.expand_dims(images[i].numpy(), 0))
            plt.title('Actual:' + class_names[labels[i].numpy()] + '\nPredicted:{} {:.2f}%'.format(
                class_names[np.argmax(prediction)], 100 * np.max(prediction)))
            plt.axis("off")
    plt.savefig(OUTPUT_DIR / 'pneumonia_sample_predictions.png', dpi=150)
    plt.close()
