from __future__ import print_function

import tensorflow as tf
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.optimizers import Adam
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
print('Class Names:', class_names)
num_classes = len(class_names)

counts = {}
for i, name in enumerate(class_names):
    folder = Path(train_dir) / name
    counts[name] = sum(1 for p in folder.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg'))
print("Training folder counts:", counts)

fig, ax = plt.subplots(figsize=(7, 4))
names = list(counts.keys())
vals = [counts[n] for n in names]
ax.bar(names, vals, color=['#4a90d9', '#7cb342', '#e57373'])
ax.set_ylabel('Images')
ax.set_title('Training set class distribution (before val split)')
for i, v in enumerate(vals):
    ax.text(i, v, str(v), ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'dataset_distribution_train.png', dpi=150)
plt.close()

n_total = float(sum(counts.values()))
class_weights = {}
for i, name in enumerate(class_names):
    w = n_total / (num_classes * float(counts[name]))
    if name in ('BACTERIAL', 'VIRAL'):
        w *= 1.15
    class_weights[i] = w
print("Class weights (inverse frequency x1.15 on sick classes):", class_weights)

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(2):
    for i in range(6):
        ax = plt.subplot(2, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i].numpy()])
        plt.axis("off")
plt.savefig(OUTPUT_DIR / 'pneumonia_train_preview.png', dpi=150)
plt.close()

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.1),
], name="data_augmentation")

try:
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(img_height, img_width, img_channels),
    )
    preprocess_input = tf.keras.applications.efficientnet.preprocess_input
    print("Using EfficientNetB0 backbone.")
except Exception as e:
    print("EfficientNetB0 weights failed ({}), falling back to MobileNetV2.".format(e))
    base = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=(img_height, img_width, img_channels),
    )
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    print("Using MobileNetV2 backbone.")

base.trainable = False

inputs = tf.keras.Input(shape=(img_height, img_width, img_channels))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(num_classes, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(learning_rate=1e-3),
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
        validation_data=val_ds,
        class_weight=class_weights,
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
