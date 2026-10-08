
import numpy as np
import tensorflow as tf
from PIL import Image

def load_rgb_image(file_or_path, img_size=128):
    img = Image.open(file_or_path).convert("RGB")
    img = img.resize((img_size, img_size))
    return np.asarray(img, dtype=np.float32) / 255.0

def make_spatial_sequence(img):
    h, w = img.shape[:2]
    hh, ww = h // 2, w // 2
    patches = [
        img[:hh, :ww],
        img[:hh, ww:],
        img[hh:, :ww],
        img[hh:, ww:],
    ]
    patches = [
        tf.image.resize(tf.convert_to_tensor(p), (img.shape[0], img.shape[1])).numpy()
        for p in patches
    ]
    return np.stack(patches).astype(np.float32)

def prepare_for_model(file_or_path, img_size=128):
    return make_spatial_sequence(load_rgb_image(file_or_path, img_size))
