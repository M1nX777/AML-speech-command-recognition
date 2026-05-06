from pathlib import Path
from tqdm import tqdm
from project_1.data.load_preprocces import (
    load_data,
    max_timeshape,
    one_hot_encoder
)
from project_1.features.feature_extraction import (
    feature_extrac_mel,
    feature_extrac_mfcc,
    file_opener,
    split_feature,
    transpose,
)
from train_model import Build_CNN_model, Build_MLP_model


def start_project():
    data_dir = Path.cwd().parent / "project_1_folder" / "data"
    data = load_data(data_dir)

    fetur_funcs = {"mfcc": feature_extrac_mfcc, "mel": feature_extrac_mel}

    fetur_DTsets = {"mfcc": {"X": [], "y": []}, "mel": {"X": [], "y": []}}

    for class_label, file_paths in tqdm(data.items()):
        for file_path in file_paths:
            features = file_opener(file_path, fetur_funcs)

            for name, value in features.items():
                fetur_DTsets[name]["X"].append(value)
                fetur_DTsets[name]["y"].append(class_label)

    max_len = max(
        max_timeshape(fetur_DTsets["mfcc"]["X"]),
        max_timeshape(fetur_DTsets["mel"]["X"]),
    )

    X_mfcc_transposed = transpose(fetur_DTsets["mfcc"]["X"])
    X_mel_transposed = transpose(fetur_DTsets["mel"]["X"])

    y_endoded, target_names = one_hot_encoder(fetur_DTsets["mfcc"]["y"])

    X_train, y_train, X_val, y_val, X_test, y_test = split_feature(
        X_mfcc_transposed, y_endoded
    )

    Build_MLP_model(
        X_train, y_train, X_val, y_val, X_test, y_test, target_names, max_len
    )

    X_train, y_train, X_val, y_val, X_test, y_test = split_feature(
        X_mel_transposed, y_endoded
    )

    Build_CNN_model(
        X_train, y_train, X_val, y_val, X_test, y_test, target_names, max_len
    )


if __name__ == "__main__":
    start_project()
