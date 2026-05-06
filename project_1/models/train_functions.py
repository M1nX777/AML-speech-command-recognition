import numpy as np
from project_1.data.load_preprocces import normalize_data, pad_time
from pathlib import Path
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from keras import regularizers
from keras.utils import plot_model
from keras.models import Sequential, Model
from keras.layers import (
    Input,
    Conv1D,
    Dense,
    Dropout,
    Concatenate,
    BatchNormalization,
    GlobalMaxPooling1D,
)


def normalize_and_pad(X, name, max_len):
    X = [normalize_data(x, name) for x in X]
    X = np.array([pad_time(x, max_len) for x in X])
    return X


def model_MLP():
    model = Sequential(
        [
            Input(shape=(32, 30)),
            BatchNormalization(),
            Dense(1024, activation="relu"),
            Dropout(0.2),
            Dense(512, activation="relu"),
            Dropout(0.2),
            Dense(400, activation="relu"),
            Dense(320, activation="relu"),
            GlobalMaxPooling1D("channels_last"),
            Dense(256, activation="relu"),
            Dense(128, activation="relu"),
            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dropout(0.2),
            Dense(8, activation="softmax"),
        ]
    )
    return model


def visualize_loss_curve(history):
    plt.plot(history.history["accuracy"], label="train acc")
    plt.plot(history.history["val_accuracy"], label="val acc")
    plt.legend()
    plt.show()

    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="val loss")
    plt.legend()
    plt.show()


def show_shape_model(model):
    return plot_model(model, show_shapes=True)


def model_CNN():
    conv_blocks = []
    inputs = Input(shape=(32, 128))
    for kernel_size in [3, 4, 5]:
        conv = Conv1D(
            32,
            kernel_size,
            padding="same",
            kernel_regularizer=regularizers.l2(0.1),
            activation="relu",
        )(inputs)
        conv = Conv1D(
            64,
            kernel_size,
            padding="same",
            kernel_regularizer=regularizers.l2(0.1),
            activation="relu",
        )(conv)
        conv = BatchNormalization()(conv)
        conv = Dropout(0.4)(conv)
        conv = GlobalMaxPooling1D()(conv)
        conv_blocks.append(conv)

    x = Concatenate()(conv_blocks)
    x = Dense(64, activation="relu")(x)
    x = Dense(32, activation="relu")(x)
    x = Dense(16, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(8, activation="softmax")(x)
    model = Model(inputs, outputs)
    return model


def get_predictions(model, X_test, y_test):
    predictions = model.predict(X_test)
    test_pred = np.argmax(predictions, axis=1)
    test_labels = np.argmax(y_test, axis=1)
    return test_pred, test_labels


def evaluate_model(y_pred, y_test, target_names):
    return print(
        classification_report(
            y_test, y_pred,
            target_names=target_names,
            digits=4)
    )


def get_confusion_matrix(y_pred, y_test):
    return print(ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, cmap="Blues"))


def save_model(model, name: str):
    path = Path.cwd().parent / "models"
    model.save(str({path}) + "_" + name + "_.keras")


def open_model(name: str):
    if name == "mfcc":
        pass
