
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from app.ml.config import IMG_SIZE, MODEL_PATH
from scripts.train import collect, parse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", required=True)
    ap.add_argument("--fake_dir", required=True)
    args = ap.parse_args()

    items = collect(args.real_dir, 0, 100000) + collect(args.fake_dir, 1, 100000)
    paths = np.array([x[0] for x in items])
    labels = np.array([x[1] for x in items], dtype=np.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(
        lambda p, y: parse(p, y, False),
        num_parallel_calls=tf.data.AUTOTUNE
    ).batch(16)

    model = tf.keras.models.load_model(MODEL_PATH)
    probs = model.predict(ds, verbose=1).ravel()
    preds = (probs >= 0.5).astype(int)

    print(classification_report(labels, preds, target_names=["REAL", "FAKE"]))
    print("ROC-AUC:", roc_auc_score(labels, probs))
    print("Confusion matrix:\n", confusion_matrix(labels, preds))

if __name__ == "__main__":
    main()
