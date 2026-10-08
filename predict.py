from pathlib import Path
import tensorflow as tf

from app.ml.config import IMG_SIZE, MODEL_PATH, LABELS
from app.ml.preprocessing import prepare_for_model
from app.ml.model import ResNetPreprocess


class DeepfakePredictor:

    def __init__(self, model_path=MODEL_PATH):

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}. Train the model first."
            )

        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "ResNetPreprocess": ResNetPreprocess
            }
        )

    def predict(self, image_file):

        x = prepare_for_model(image_file, IMG_SIZE)

        p_fake = float(
            self.model.predict(
                x[None, ...],
                verbose=0
            )[0][0]
        )

        label_id = int(p_fake >= 0.5)

        confidence = (
            p_fake if label_id
            else 1.0 - p_fake
        )

        return {
            "label": LABELS[label_id],
            "fake_probability": p_fake,
            "real_probability": 1.0 - p_fake,
            "confidence": confidence
        }