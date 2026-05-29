from datetime import datetime
from pathlib import Path
from typing import Sequence
from IPython.display import Image

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import regularizers
from keras.callbacks import History, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.layers import (
    AveragePooling2D,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Dense,
    Dropout,
    GlobalMaxPooling1D,
    GlobalMaxPooling2D,
    Input,
)
from keras.models import Model, Sequential
from keras.utils import plot_model
from project_1.data.load_preprocces import normalize_data, pad_time
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import KFold


def save_callbacks(path: Path) -> ModelCheckpoint:
    save_mod = ModelCheckpoint(
        filepath=path,
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        save_freq="epoch",
    )
    return save_mod


def cross_callbacks(patience_lr: int) -> ReduceLROnPlateau:
    reduce_lr_loss = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.1,
        patience=patience_lr,
        verbose=1,
        min_delta=1e-4,
        mode="min",
    )
    return reduce_lr_loss


def cross_validation(
    n_folds: int, X: Sequence[np.ndarray] | np.ndarray, y: np.ndarray, name: str
) -> list[list[float]]:
    acc_per_fold = []
    loss_per_fold = []
    kf = KFold(n_splits=n_folds)
    for i, (train, val) in enumerate(kf.split(X, y)):
        log_name = f"{name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        keras.backend.clear_session()

        X_train_cut = X[train]
        y_train_cut = y[train]

        X_valid_cut = X[val]
        y_valid_cut = y[val]

        tensor = TensorBoard(log_dir=f"logs/fold_{i + 1}/{log_name}")
        callback = cross_callbacks(3)

        F1 = keras.metrics.F1Score(average="macro")
        optimize = keras.optimizers.AdamW()
        model = model_MLP() if name == "mfcc" else model_CNN()

        model.compile(
            optimizer=optimize,
            loss="categorical_crossentropy",
            metrics=["accuracy", F1],
        )
        print(
            "------------------------------------------------------------------------"
        )
        print(f"Training for fold {i + 1} ...")
        model.fit(
            X_train_cut,
            y_train_cut,
            epochs=20 if name == "mfcc" else 35,
            validation_data=(X_valid_cut, y_valid_cut),
            callbacks=[callback, tensor],
        )

        scores = model.evaluate(X_valid_cut, y_valid_cut)
        print(
            f"Score for fold {i}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%"
        )
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])

    return [acc_per_fold, loss_per_fold]


def normalize_and_pad(
    X: Sequence[np.ndarray] | np.ndarray,
    name: str,
    max_len: int,
) -> np.ndarray | Sequence[np.ndarray]:
    normalized = [normalize_data(x, name) for x in X]
    padded = np.array([pad_time(x, max_len) for x in normalized])
    return padded


def model_MLP() -> Model:

    model = Sequential(
        [
            Input(shape=(87, 30)),
            BatchNormalization(),
            Dense(256, activation="relu"),
            Dense(256, activation="relu"),
            Dense(256, activation="relu"),
            GlobalMaxPooling1D("channels_last"),
            Dense(256, activation="relu"),
            Dropout(0.2),
            Dense(8, activation="softmax"),
        ]
    )
    return model


def visualize_loss_curve(history: History) -> None:
    plt.plot(history.history["accuracy"], label="train acc")
    plt.plot(history.history["val_accuracy"], label="val acc")
    plt.legend()
    plt.show()

    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="val loss")
    plt.legend()
    plt.show()


def show_shape_model(model: Model) -> Image | None:
    return plot_model(model, show_shapes=True)


def model_CNN() -> Model:
    conv_blocks = []
    inputs = Input(shape=(87, 128, 1))
    for kernel_size in [(3, 5), (3, 7), (3, 9), (3, 11)]:
        conv = Conv2D(
            32,
            kernel_size,
            padding="same",
            kernel_regularizer=regularizers.l2(0.01),
            activation="relu",
        )(inputs)
        conv = AveragePooling2D(pool_size=(2, 2))(conv)
        conv = Conv2D(
            64,
            kernel_size,
            padding="same",
            kernel_regularizer=regularizers.l2(0.01),
            activation="relu",
        )(conv)
        conv = AveragePooling2D(pool_size=(2, 2))(conv)
        conv = Conv2D(
            128,
            kernel_size,
            padding="same",
            kernel_regularizer=regularizers.l2(0.01),
            activation="relu",
        )(conv)
        conv = BatchNormalization()(conv)
        conv = GlobalMaxPooling2D()(conv)
        conv_blocks.append(conv)
    x = Concatenate()(conv_blocks)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.2)(x)
    outputs = Dense(8, activation="softmax")(x)
    model = Model(inputs, outputs)
    conv_blocks = []
    return model


def get_predictions(
    model: Model,
    X_test: Sequence[np.ndarray] | np.ndarray,
    y_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    predictions = model.predict(X_test)
    test_pred = np.argmax(predictions, axis=1)
    test_labels = np.argmax(y_test, axis=1)
    return test_pred, test_labels


def evaluate_model(
    y_pred: np.ndarray,
    y_test: np.ndarray,
    target_names: list[str],
) -> None:
    y_true_list = np.asarray(y_test).tolist()
    y_pred_list = np.asarray(y_pred).tolist()
    print(
        classification_report(
            y_true_list, y_pred_list, target_names=target_names, digits=4
        )
    )


def get_confusion_matrix(y_pred: np.ndarray, y_test: np.ndarray) -> None:
    print(ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues"))
