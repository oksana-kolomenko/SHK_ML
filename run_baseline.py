import numpy as np

from csv_saver import save_results_to_csv
from data_prep import mimic_nom_features
from data_preps import load_features, load_labels
from helpers import (hgbc_rte, hgbc, logistic_regression, lr_rte)

from values import DatasetName


def run_models_on_table_data():
    # === CYBERSECURITY ===
    """
    dataset = DatasetName.CYBERSECURITY.value
    X = load_features(file_path="X_cybersecurity_intrusion_data.csv")
    y = load_labels(file_path="y_cybersecurity_intrusion_data.csv")

    nominal_features = [
        'encryption_used',
        'browser_type',
        'protocol_type'
    ]
    """


    # === LUNG DISEASE ===
    """
    dataset = DatasetName.LUNG_DISEASE.value
    X = load_features(file_path="X_lung_disease_data.csv")
    y = load_labels(file_path="y_lung_disease_data.csv")
    
    nominal_features = [
        "Gender",
        "Smoking Status",
        "Disease Type",
        "Treatment Type"
    ]
    """

    # === MIMIC ===
    task = "task_1"
    dataset = DatasetName.MIMIC_1.value

    X_train = load_features(f"mimic_data/X_train_{task}.csv")
    X_test = load_features(f"mimic_data/X_test_{task}.csv")

    y_train = load_labels(f"mimic_data/y_train_{task}.csv")
    y_test = load_labels(f"mimic_data/y_test_{task}.csv")

    nominal_features = mimic_nom_features(X=X_train)

    # 1. logistic regression
    (log_reg_dataset, log_reg_ml_method, log_reg_emb_method, log_reg_conc, log_reg_best_params,
     log_reg_pca_comp, log_reg_train_score, log_reg_test_scores) = \
        logistic_regression(dataset_name=dataset,
                            nominal_features=nominal_features,
                            X_train=X_train,
                            X_test=X_test,
                            y_train=y_train,
                            y_test=y_test)

    save_results_to_csv(
        dataset_name=log_reg_dataset,
        ml_method=log_reg_ml_method,
        emb_method=log_reg_emb_method,
        pca_n_comp=log_reg_pca_comp,
        best_params=log_reg_best_params,
        concatenation=log_reg_conc,
        is_train=True,
        metrics=log_reg_train_score,
        output_file=f"{dataset}_log_reg_train.csv")

    save_results_to_csv(
        dataset_name=log_reg_dataset,
        ml_method=log_reg_ml_method,
        emb_method=log_reg_emb_method,
        pca_n_comp=log_reg_pca_comp,
        best_params=log_reg_best_params,
        concatenation=log_reg_conc,
        is_train=False,
        metrics=log_reg_test_scores,
        output_file=f"{dataset}_log_reg_test.csv")

    # 2. log reg + random trees embedding
    (lr_rt_dataset, lr_rt_ml_method, lr_rt_emb_method, lr_rt_concatenation, lr_rte_best_params, lr_rte_pca,
     lr_rte_train_score, lr_rte_test_scores) = \
        lr_rte(dataset_name=dataset,
               nominal_features=nominal_features,
               pca=False,
               X_train=X_train,
               X_test=X_test,
               y_train=y_train,
               y_test=y_test)

    save_results_to_csv(
        dataset_name=lr_rt_dataset,
        concatenation=lr_rt_concatenation,
        emb_method=lr_rt_emb_method,
        is_train=True,
        metrics=lr_rte_train_score,
        ml_method=lr_rt_ml_method,
        best_params=lr_rte_best_params,
        pca_n_comp=lr_rte_pca,
        output_file=f"{dataset}_lr_rte_train.csv")

    save_results_to_csv(
        dataset_name=lr_rt_dataset,
        concatenation=lr_rt_concatenation,
        emb_method=lr_rt_emb_method,
        is_train=False,
        metrics=lr_rte_test_scores,
        ml_method=lr_rt_ml_method,
        best_params=lr_rte_best_params,
        pca_n_comp=lr_rte_pca,
        output_file=f"{dataset}_lr_rte_test.csv")

    # 3. hgbc (no embedding)
    """
    hgbc_dataset, hgbc_ml_method, hgbc_emb_method, conc, hgbc_best_params, hgbc_train_score, hgbc_test_scores = \
        hgbc(dataset_name=dataset, X=X, y=y, nominal_features=nominal_features)

    save_results_to_csv(
        dataset_name=hgbc_dataset,
        ml_method=hgbc_ml_method,
        emb_method=hgbc_emb_method,
        pca_n_comp="none",
        best_params=hgbc_best_params,
        concatenation=conc,
        is_train=True,
        metrics=hgbc_train_score,
        output_file=f"{dataset}_hgbc_train.csv")

    save_results_to_csv(
        dataset_name=hgbc_dataset,
        ml_method=hgbc_ml_method,
        emb_method=hgbc_emb_method,
        pca_n_comp="none",
        best_params=hgbc_best_params,
        concatenation=conc,
        is_train=False,
        metrics=hgbc_test_scores,
        output_file=f"{dataset}_hgbc_test.csv")

    # 4. random trees embedding + hgbc
    (hgbc_rt_dataset, hgbc_rt_ml_method, hgbc_rt_emb_method, hgbc_rte_conc, hgbc_rte_best_params,
     hgbc_rt_emb_train_score, hgbc_rt_emb_test_scores) = \
        hgbc_rte(dataset_name=dataset, X=X, y=y,
                 nominal_features=nominal_features)

    save_results_to_csv(
        dataset_name=hgbc_rt_dataset,
        ml_method=hgbc_rt_ml_method,
        emb_method=hgbc_rt_emb_method,
        pca_n_comp="none",
        best_params=hgbc_rte_best_params,
        concatenation=hgbc_rte_conc,
        is_train=True,
        metrics=hgbc_rt_emb_train_score,
        output_file=f"{dataset}_HGBC_rte_train.csv")

    save_results_to_csv(
        dataset_name=hgbc_rt_dataset,
        ml_method=hgbc_rt_ml_method,
        emb_method=hgbc_rt_emb_method,
        pca_n_comp="none",
        best_params=hgbc_rte_best_params,
        concatenation=hgbc_rte_conc,
        is_train=False,
        metrics=hgbc_rt_emb_test_scores,
        output_file=f"{dataset}_HGBC_rte_test.csv")
    """
