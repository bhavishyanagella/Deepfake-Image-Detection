import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50


@tf.keras.utils.register_keras_serializable()
class ResNetPreprocess(layers.Layer):

    def call(self, inputs):
        # Input is normalized to [0, 1]
        x = inputs * 255.0

        # ResNet50 preprocessing
        return tf.keras.applications.resnet50.preprocess_input(x)

    def compute_output_shape(self, input_shape):
        return input_shape


def build_model(img_size=128, seq_len=4):

    inp = layers.Input(
        shape=(seq_len, img_size, img_size, 3),
        name="image_regions"
    )

    # Pretrained ResNet50
    base = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(img_size, img_size, 3)
    )

    # Freeze ResNet50 for much faster CPU training
    for layer in base.layers:
        layer.trainable = False

    # Preprocess each of the 4 regions
    x = layers.TimeDistributed(
        ResNetPreprocess(),
        name="resnet_preprocess"
    )(inp)

    # CNN feature extraction
    x = layers.TimeDistributed(
        base,
        name="cnn"
    )(x)

    # Convert CNN feature maps into feature vectors
    x = layers.TimeDistributed(
        layers.GlobalAveragePooling2D()
    )(x)

    # Process 4 spatial regions as a sequence
    x = layers.LSTM(
        128,
        name="lstm"
    )(x)

    x = layers.Dropout(0.4)(x)

    x = layers.Dense(
        64,
        activation="relu"
    )(x)

    x = layers.Dropout(0.3)(x)

    x = layers.Dense(
        32,
        activation="relu"
    )(x)

    out = layers.Dense(
        1,
        activation="sigmoid",
        name="deepfake_probability"
    )(x)

    return Model(
        inp,
        out,
        name="ResNet50_CNN_LSTM_Deepfake"
    )


def compile_model(model):

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ]
    )

    return model