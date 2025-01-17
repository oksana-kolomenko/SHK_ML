import numpy as np

from csv_saver import save_results_to_csv
from helpers import (load_labels, load_features, logistic_regression,)
from pca_methods import find_best_n_components_and_save_csv
from values import Dataset


def run():
    posttrauma_dataset = Dataset.POSTTRAUMA.value

    # load features and labels
    X_posttrauma = load_features()
    y_posttrauma = load_labels()

    # define nominal features
    nominal_features = [
        'gender_birth',
        'ethnic_group',
        'education_age',
        'working_at_baseline',
        'penetrating_injury'
    ]

    n_components_range = range(1, 11)  # Test 1 to 10 components

    # Run the optimization and save results
    results = find_best_n_components_and_save_csv(
        dataset_name=posttrauma_dataset,
        X=X_posttrauma,
        y=y_posttrauma,
        nominal_features=nominal_features,
        n_splits=3,
        n_components_range=n_components_range,
    )

    # Display results
    print("Best n_components:", results["Best n_components"])
    print("Best Metrics:", results["Best Metrics"])


"""    # 1. logistic regression (no embedding), dataset: posttrauma
    log_reg_dataset, log_reg_ml_method, log_reg_emb_method, log_reg_train_metrics, log_reg_test_metrics = \
        logistic_regression(dataset_name=posttrauma_dataset, X=X_posttrauma, y=y_posttrauma,  # test
                            nominal_features=nominal_features)

    save_results_to_csv(
        emb_method=log_reg_emb_method,
        dataset_name=log_reg_dataset,
        ml_method=log_reg_ml_method,
        output_file="no_emb_log_reg_train.csv",
        metrics=log_reg_train_metrics,
        is_train=True
    )

    save_results_to_csv(
        emb_method=log_reg_emb_method,
        dataset_name=log_reg_dataset,
        ml_method=log_reg_ml_method,
        output_file="no_emb_log_reg_test.csv",
        metrics=log_reg_test_metrics,
        is_train=False
    )"""
