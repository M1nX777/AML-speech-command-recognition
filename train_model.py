import keras
from keras.callbacks import EarlyStopping
from project_1.models.train_functions import (
    model_CNN,
    model_MLP,
    normalize_and_pad,
    visualize_loss_curve,
    show_shape_model,
    get_predictions,
    get_confusion_matrix,
    evaluate_model,
)

keras.utils.set_random_seed(42)


def Build_MLP_model(
    X_train, y_train, X_val, y_val, X_test, y_test, target_names, max_len
):
    name = "mfcc"
    X_train = normalize_and_pad(X_train, name, max_len)
    X_val = normalize_and_pad(X_val, name, max_len)
    X_test = normalize_and_pad(X_test, name, max_len)

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=2, restore_best_weights=True
    )
    model = model_MLP()
    optimize = keras.optimizers.AdamW(learning_rate=1e-5, weight_decay=1e-2)

    model.compile(
        optimizer=optimize,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=100,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
    )
    visualize_loss_curve(history)
    show_shape_model(model)
    y_pred, test_lables = get_predictions(model, X_test, y_test)
    evaluate_model(y_pred, test_lables, target_names)
    get_confusion_matrix(y_pred, test_lables)


def Build_CNN_model(
    X_train, y_train, X_val, y_val, X_test, y_test, target_names, max_len
):
    name = "mel"
    X_train = normalize_and_pad(X_train, name, max_len)
    X_val = normalize_and_pad(X_val, name, max_len)
    X_test = normalize_and_pad(X_test, name, max_len)

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=2, restore_best_weights=True
    )

    model = model_CNN()
    optimize_cnn = keras.optimizers.AdamW(learning_rate=1e-5)

    model.compile(
        optimizer=optimize_cnn,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=200,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
    )

    visualize_loss_curve(history)
    show_shape_model(model)
    y_pred, test_lables = get_predictions(model, X_test, y_test)
    evaluate_model(y_pred, test_lables, target_names)
    get_confusion_matrix(y_pred, test_lables)
