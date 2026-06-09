from pathlib import Path
from typing import Sequence

import keras
import numpy as np
from keras.callbacks import EarlyStopping
from keras.metrics import F1Score
from keras.optimizers import AdamW
from project_1.features.feature_extraction import add_gaussian_noise
from project_1.models.train_functions import (
    cross_validation,
    evaluate_model,
    get_confusion_matrix,
    get_predictions,
    model_CNN,
    model_MLP,
    save_callbacks,
    show_shape_model,
    visualize_loss_curve,
)

keras.utils.set_random_seed(42)


def Build_MLP_model(
    X_train_val: Sequence[np.ndarray],
    y_train_val: np.ndarray,
    X_test: Sequence[np.ndarray],
    y_test: np.ndarray,
    target_names: list[str],
) -> None:
    name = "mfcc"
    n_folds = 10
    X_train_val = add_gaussian_noise(
        X_train_val,
        noise_factor=0.5,
        random_state=42
    )
    mean_acc, std_dev = cross_validation(n_folds, X_train_val, y_train_val, name)

    print(
        f"The mean accuracy of the Mel-frequency cepstrum model accros all folds is:\n {np.mean(mean_acc)}",
        f"with a standard deviation of: {np.std(std_dev)}",
    )

    print("Train final mfcc model after tunning")
    optimize = AdamW(learning_rate=1e-6)
    f1_score = F1Score(average="macro")
    stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

    model = model_MLP()

    model.compile(
        optimizer=optimize,
        loss="categorical_crossentropy",
        metrics=["accuracy", f1_score],
    )

    save_name = name + ".keras"
    path = Path.cwd().parent / "project_1_folder" / "models" / save_name
    save = save_callbacks(path)

    history = model.fit(
        X_train_val,
        y_train_val,
        epochs=100,
        validation_split=0.1,
        callbacks=[stop, save],
    )

    visualize_loss_curve(history)
    show_shape_model(model)
    y_pred, test_lables = get_predictions(model, X_test, y_test)
    evaluate_model(y_pred, test_lables, target_names)
    get_confusion_matrix(y_pred, test_lables)


def Build_CNN_model(
    X_train_val: Sequence[np.ndarray],
    y_train_val: np.ndarray,
    X_test: Sequence[np.ndarray] | np.ndarray,
    y_test: np.ndarray,
    target_names: list[str],
) -> None:
    name = "mel"
    n_folds = 10
    X_train_val = add_gaussian_noise(
        X_train_val,
        noise_factor=0.5,
        random_state=42
    )
    X_train_val = X_train_val[..., np.newaxis]

    mean_acc, std_dev = model_CNN()
    history = cross_validation(n_folds, X_train_val, y_train_val, name)

    print(
        f"The mean accuracy of the Mel Spectrogram model accros all folds is:\n {np.mean(mean_acc)}",
        f"with a standard deviation of: {np.std(std_dev)}",
    )

    print("Train final mel model after tunning")
    optimize = AdamW(learning_rate=1e-5, weight_decay=1e-2)
    f1_score = F1Score(average="macro")
    stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

    model = model_CNN()

    model.compile(
        optimizer=optimize,
        loss="categorical_crossentropy",
        metrics=["accuracy", f1_score],
    )
    save_name = name + ".keras"
    path = Path.cwd().parent / "project_1_folder" / "models" / save_name
    save = save_callbacks(path)

    history = model.fit(
        X_train_val,
        y_train_val,
        epochs=100,
        validation_split=0.1,
        callbacks=[stop, save],
    )

    visualize_loss_curve(history)
    show_shape_model(model)
    y_pred, test_lables = get_predictions(model, X_test, y_test)
    evaluate_model(y_pred, test_lables, target_names)
    get_confusion_matrix(y_pred, test_lables)
