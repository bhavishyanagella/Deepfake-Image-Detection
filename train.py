
import argparse
import random
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks

from app.ml.config import IMG_SIZE, SEQ_LEN, MODEL_PATH
from app.ml.model import build_model, compile_model

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
EXTS = {".jpg", ".jpeg", ".png", ".webp"}

def collect(folder, label, limit):
    paths = [p for p in Path(folder).rglob("*") if p.suffix.lower() in EXTS]
    random.shuffle(paths)
    return [(str(p), label) for p in paths[:limit]]

def parse(path, label, training=False):
    raw = tf.io.read_file(path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32) / 255.0

    if training:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.08)
        img = tf.image.random_contrast(img, 0.9, 1.1)

    h = IMG_SIZE // 2
    w = IMG_SIZE // 2
    patches = tf.stack([
        img[:h, :w],
        img[:h, w:],
        img[h:, :w],
        img[h:, w:]
    ])
    patches = tf.image.resize(patches, (IMG_SIZE, IMG_SIZE))
    return patches, tf.cast(label, tf.float32)

def make_ds(items, training):
    paths = np.array([x[0] for x in items])
    labels = np.array([x[1] for x in items], dtype=np.float32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(min(len(items), 2000), seed=SEED)
    ds = ds.map(
        lambda p, y: parse(p, y, training),
        num_parallel_calls=1
    )
    return ds.batch(4).prefetch(1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", required=True)
    ap.add_argument("--fake_dir", required=True)
    ap.add_argument("--per_class", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()

    real = collect(args.real_dir, 0, args.per_class)
    fake = collect(args.fake_dir, 1, args.per_class)
    items = real + fake

    train_items, val_items = train_test_split(
        items,
        test_size=0.2,
        random_state=SEED,
        stratify=[x[1] for x in items]
    )

    print(f"Train: {len(train_items)} | Validation: {len(val_items)}")
    model = compile_model(build_model(IMG_SIZE, SEQ_LEN))

    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    cbs = [
        callbacks.ModelCheckpoint(
            str(MODEL_PATH), monitor="val_auc",
            mode="max", save_best_only=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2,
            patience=2, min_lr=1e-7, verbose=1
        ),
        callbacks.EarlyStopping(
            monitor="val_auc", mode="max",
            patience=5, restore_best_weights=True, verbose=1
        )
    ]

    model.fit(
        make_ds(train_items, True),
        validation_data=make_ds(val_items, False),
        epochs=args.epochs,
        callbacks=cbs
    )
    print("Saved:", MODEL_PATH)

if __name__ == "__main__":
    main()
