# Chest X-ray: BACTERIAL / NORMAL / VIRAL

from __future__ import print_function

import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import matplotlib

if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
DATA_ROOT = SCRIPT_DIR / "chest_xray"
TRAIN_DIR = str(DATA_ROOT / "train")
TEST_DIR = str(DATA_ROOT / "test")
MODEL_PATH = OUTPUT_DIR / "best_pneumonia.keras"

RNG_SEED = 123
batch_size = 16
epochs_head = 25
epochs_finetune = 12
img_width = img_height = 224
img_channels = 3
fit = True
do_finetune = True

# Avoid HTTP 403 when downloading ImageNet weights
BACKBONE_LAYER_NAME = "transfer_backbone"


def patch_weights_download_user_agent():
    import urllib.request

    opener = urllib.request.build_opener()
    opener.addheaders = [
        (
            "User-Agent",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        ),
    ]
    urllib.request.install_opener(opener)


def build_model(num_classes, trainable_backbone=False):
    # EfficientNetB0 if weights OK, else MobileNetV2 (GAP + dense head)
    base = None
    preprocess_input = None
    try:
        base = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=(img_height, img_width, img_channels),
            name=BACKBONE_LAYER_NAME,
        )
        preprocess_input = tf.keras.applications.efficientnet.preprocess_input
        print("Backbone: EfficientNetB0 (ImageNet weights).")
    except Exception as e:
        print("EfficientNet weights unavailable ({}). Using MobileNetV2 instead.".format(e))
        base = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights="imagenet",
            input_shape=(img_height, img_width, img_channels),
            name=BACKBONE_LAYER_NAME,
        )
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
        print("Backbone: MobileNetV2 (ImageNet weights).")

    base.trainable = trainable_backbone

    aug = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.06),
            tf.keras.layers.RandomZoom(0.08),
            tf.keras.layers.RandomContrast(0.12),
        ],
        name="data_augmentation",
    )

    inputs = tf.keras.Input(shape=(img_height, img_width, img_channels))
    x = aug(inputs)
    x = preprocess_input(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dropout(0.35, name="head_dropout1")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="head_dense")(x)
    x = tf.keras.layers.Dropout(0.25, name="head_dropout2")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="head_softmax")(x)
    m = tf.keras.Model(inputs, outputs, name="chest_xray_transfer")
    m._gradcam_preprocess = preprocess_input
    return m, base


def infer_preprocess_from_backbone(backbone):
    # Choose EfficientNet vs MobileNet preprocess after load.
    names = {layer.name for layer in backbone.layers}
    if "stem_conv" in names:
        return tf.keras.applications.efficientnet.preprocess_input
    return tf.keras.applications.mobilenet_v2.preprocess_input


def make_gradcam_heatmap(img_batch, model, backbone, pred_index=None):
    # GradCAM via GradientTape (Keras 3 nested model workaround).
    aug = model.get_layer("data_augmentation")
    preproc = getattr(model, "_gradcam_preprocess", None)
    if preproc is None:
        preproc = infer_preprocess_from_backbone(backbone)

    gap = model.get_layer("gap")
    drop1 = model.get_layer("head_dropout1")
    dense = model.get_layer("head_dense")
    drop2 = model.get_layer("head_dropout2")
    softmax = model.get_layer("head_softmax")

    with tf.GradientTape() as tape:
        x = aug(img_batch, training=False)
        x = preproc(x)
        conv_out = backbone(x, training=False)
        tape.watch(conv_out)
        h = gap(conv_out)
        h = drop1(h, training=False)
        h = dense(h)
        h = drop2(h, training=False)
        preds = softmax(h)
        if pred_index is None:
            pred_index = int(np.argmax(preds[0].numpy()))
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_out)
    pooled = tf.reduce_mean(grads, axis=(1, 2))
    conv_out = conv_out[0]
    heatmap = tf.reduce_sum(tf.cast(pooled, conv_out.dtype) * conv_out, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    m = tf.reduce_max(heatmap)
    heatmap = heatmap / (m + 1e-10)
    return heatmap.numpy(), pred_index


def overlay_gradcam(img_uint8, heatmap, alpha=0.45):
    try:
        import cv2

        img = img_uint8.astype(np.float32)
        hm = np.uint8(255 * heatmap)
        hm = cv2.resize(hm, (img.shape[1], img.shape[0]))
        hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
        hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
        return (alpha * hm + (1 - alpha) * img).astype(np.uint8)
    except ImportError:
        h, w = img_uint8.shape[0], img_uint8.shape[1]
        hm = tf.image.resize(
            heatmap.reshape(1, heatmap.shape[0], heatmap.shape[1], 1),
            (h, w),
            method="bilinear",
        ).numpy().squeeze()
        jet = plt.cm.jet(np.clip(hm, 0, 1))[:, :, :3]
        imgf = img_uint8.astype(np.float32) / 255.0
        blend = alpha * jet + (1 - alpha) * imgf
        return (np.clip(blend, 0, 1) * 255.0).astype(np.uint8)


def main(
    fit_override=None,
    do_finetune_override=None,
):
    global fit, do_finetune
    if fit_override is not None:
        fit = fit_override
    if do_finetune_override is not None:
        do_finetune = do_finetune_override

    np.random.seed(RNG_SEED)
    tf.random.set_seed(RNG_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    patch_weights_download_user_agent()

    for gpu in tf.config.experimental.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    train_ds, val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        TRAIN_DIR,
        seed=RNG_SEED,
        validation_split=0.2,
        subset="both",
        image_size=(img_height, img_width),
        batch_size=batch_size,
        labels="inferred",
        shuffle=True,
    )

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        TEST_DIR,
        seed=RNG_SEED,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        labels="inferred",
        shuffle=False,
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print("Class names (label order):", class_names)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    test_ds = test_ds.cache().prefetch(AUTOTUNE)

    counts = {}
    for i, name in enumerate(class_names):
        folder = Path(TRAIN_DIR) / name
        counts[name] = sum(
            1 for p in folder.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")
        )
    print("Training folder counts:", counts)
    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(counts.keys())
    vals = [counts[n] for n in names]
    ax.bar(names, vals, color=["#4a90d9", "#7cb342", "#e57373"])
    ax.set_ylabel("Images")
    ax.set_title("Training set class distribution (before val split)")
    for i, v in enumerate(vals):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "dataset_distribution_train.png", dpi=150)
    plt.close()

    n_total = float(sum(counts.values()))
    class_weights = {}
    for i, name in enumerate(class_names):
        w = n_total / (num_classes * float(counts[name]))
        if name in ("BACTERIAL", "VIRAL"):
            w *= 1.15
        class_weights[i] = w
    print("Class weights (inverse frequency x1.15 on sick classes):", class_weights)

    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    for images, labels in train_ds.take(1):
        for i, ax in enumerate(axes.flat):
            if i < min(6, images.shape[0]):
                ax.imshow(images[i].numpy().astype("uint8"))
                ax.set_title(class_names[int(labels[i])])
            ax.axis("off")
    plt.suptitle("Training batch preview")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pneumonia_train_preview.png", dpi=150)
    plt.close()

    model, backbone = build_model(num_classes, trainable_backbone=False)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    save_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )
    early_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=7,
        restore_best_weights=True,
        verbose=1,
    )
    lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    )

    history = None
    t_train = None

    if fit:
        t0 = time.perf_counter()
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs_head,
            class_weight=class_weights,
            callbacks=[save_callback, early_callback, lr_reduce],
            verbose=1,
        )
        t1 = time.perf_counter()

        if do_finetune:
            backbone.trainable = True
            freeze_until = int(len(backbone.layers) * 0.65)
            for layer in backbone.layers[:freeze_until]:
                layer.trainable = False
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            hist2 = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs_head + epochs_finetune,
                initial_epoch=len(history.history["loss"]),
                class_weight=class_weights,
                callbacks=[save_callback, early_callback, lr_reduce],
                verbose=1,
            )
            for k in history.history:
                history.history[k].extend(hist2.history[k])
            t1 = time.perf_counter()

        t_train = t1 - t0
        with open(OUTPUT_DIR / "training_wall_time_sec.txt", "w", encoding="utf-8") as f:
            f.write("Total training wall time (seconds): {:.1f}\n".format(t_train))
            f.write("Total training wall time (minutes): {:.2f}\n".format(t_train / 60.0))
        print("\n[Q1] Total training wall time: {:.1f} s ({:.2f} min)\n".format(t_train, t_train / 60.0))
        model.save(str(MODEL_PATH))
    else:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                'No saved model at "{}". Train first with fit=True.'.format(MODEL_PATH)
            )
        model = tf.keras.models.load_model(str(MODEL_PATH))
        try:
            backbone = model.get_layer(BACKBONE_LAYER_NAME)
        except ValueError:
            backbone = model.get_layer("efficientnet_body")

    test_scores = model.evaluate(test_ds, verbose=0)
    print("Test loss:", test_scores[0])
    print("Test accuracy:", test_scores[1])

    if history is not None:
        import json
        with open(OUTPUT_DIR / "training_history.json", "w") as f:
            json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(history.history["accuracy"], label="train")
        ax.plot(history.history["val_accuracy"], label="validation")
        ax.set_title("Model accuracy (train vs val - Q3 overfitting check)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "pneumonia_accuracy.png", dpi=150)
        plt.close()

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(history.history["loss"], label="train")
        ax.plot(history.history["val_loss"], label="validation")
        ax.set_title("Model loss (train vs val - Q3 overfitting check)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "pneumonia_loss.png", dpi=150)
        plt.close()

    probs = model.predict(test_ds, verbose=0)
    y_true = np.concatenate([labels.numpy() for _, labels in test_ds], axis=0)
    y_pred = np.argmax(probs, axis=1)

    print("\n[Q7] Classification report (test set) - precision / recall / F1:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            digits=4,
        )
    )

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    tick_marks = np.arange(len(class_names))
    ax.set(
        xticks=tick_marks,
        yticks=tick_marks,
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Chest X-ray test confusion matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pneumonia_confusion_matrix.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for images, labels in test_ds.take(1):
        for i, ax in enumerate(axes.flat):
            if i >= min(6, images.shape[0]):
                ax.axis("off")
                continue
            img = images[i].numpy()
            pred = model.predict(np.expand_dims(img, 0), verbose=0)[0]
            pred_idx = int(np.argmax(pred))
            ax.imshow(img.astype("uint8"))
            ax.set_title(
                "Actual: {}\nPred: {} ({:.1f}%)".format(
                    class_names[int(labels[i])],
                    class_names[pred_idx],
                    100.0 * float(pred[pred_idx]),
                ),
                fontsize=9,
            )
            ax.axis("off")
    plt.suptitle("Sample test predictions")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pneumonia_sample_predictions.png", dpi=150)
    plt.close()

    try:
        for images, labels in test_ds.take(1):
            for i in range(min(3, images.shape[0])):
                img = images[i : i + 1]
                hm, pidx = make_gradcam_heatmap(img, model, backbone)
                raw = img[0].numpy().astype(np.uint8)
                vis = overlay_gradcam(raw, hm)
                fig, ax = plt.subplots(1, 2, figsize=(8, 4))
                ax[0].imshow(raw)
                ax[0].set_title("Input")
                ax[0].axis("off")
                ax[1].imshow(vis)
                ax[1].set_title("GradCAM (pred: {})".format(class_names[pidx]))
                ax[1].axis("off")
                plt.tight_layout()
                plt.savefig(OUTPUT_DIR / "gradcam_sample_{}.png".format(i), dpi=150)
                plt.close()
            break
    except Exception as e:
        print("GradCAM figures skipped:", e)

    model.summary()
    print('\nSaved figures, timing, and model to "{}"'.format(OUTPUT_DIR))


if __name__ == "__main__":
    main()
