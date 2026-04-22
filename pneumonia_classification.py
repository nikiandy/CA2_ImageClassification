from __future__ import print_function

import tensorflow as tf
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.optimizers import Adam
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

batch_size = 16
num_classes = 3
epochs_head = 15
epochs_finetune = 10
img_width = 224
img_height = 224
img_channels = 3
fit = True
do_finetune = True
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
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    labels='inferred',
    shuffle=False)

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
    monitor='val_accuracy', mode='max', patience=7, restore_best_weights=True, verbose=1)
save_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=str(OUTPUT_DIR / "best_pneumonia.keras"),
    monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)
lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

history = None

if fit:
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_head,
        class_weight=class_weights,
        callbacks=[save_callback, earlystop_callback, lr_reduce],
        verbose=1)

    if do_finetune:
        base.trainable = True
        freeze_until = int(len(base.layers) * 0.65)
        for layer in base.layers[:freeze_until]:
            layer.trainable = False
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
        hist2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs_head + epochs_finetune,
            initial_epoch=len(history.history['loss']),
            class_weight=class_weights,
            callbacks=[save_callback, earlystop_callback, lr_reduce],
            verbose=1)
        for k in history.history:
            history.history[k].extend(hist2.history[k])

    model.save(str(OUTPUT_DIR / "best_pneumonia.keras"))
else:
    model = tf.keras.models.load_model(str(OUTPUT_DIR / "best_pneumonia.keras"))

score = model.evaluate(test_ds, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

if history is not None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history.history['accuracy'], label='train')
    ax.plot(history.history['val_accuracy'], label='validation')
    ax.set_title('Model accuracy (train vs val)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'pneumonia_accuracy.png', dpi=150)
    plt.close()

probs = model.predict(test_ds, verbose=0)
y_true = np.concatenate([labels.numpy() for _, labels in test_ds], axis=0)
y_pred = np.argmax(probs, axis=1)

print("\nClassification report (test set):")
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
tick_marks = np.arange(len(class_names))
ax.set(
    xticks=tick_marks,
    yticks=tick_marks,
    xticklabels=class_names,
    yticklabels=class_names,
    ylabel='True label',
    xlabel='Predicted label',
    title='Chest X-ray test confusion matrix',
)
plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
thresh = cm.max() / 2.0 if cm.size else 0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha='center', va='center',
                color='white' if cm[i, j] > thresh else 'black')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'pneumonia_confusion_matrix.png', dpi=150)
plt.close()

test_batch = test_ds.take(1)
plt.figure(figsize=(10, 10))
for images, labels in test_batch:
    for i in range(6):
        ax = plt.subplot(2, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        prediction = model.predict(tf.expand_dims(images[i].numpy(), 0), verbose=0)
        plt.title('Actual:' + class_names[labels[i].numpy()] + '\nPredicted:{} {:.2f}%'.format(
            class_names[np.argmax(prediction)], 100 * np.max(prediction)))
        plt.axis("off")
plt.savefig(OUTPUT_DIR / 'pneumonia_sample_predictions.png', dpi=150)
plt.close()
