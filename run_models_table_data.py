import numpy as np

from csv_saver import save_results_to_csv
from data_preps import load_features, load_labels
from helpers import (hgbc_rte, hgbc, logistic_regression, lr_rte)

from values import DatasetName


def run_models_on_table_data():
    """
    dataset = DatasetName.POSTTRAUMA.value
    nominal_features = [
        'gender_birth',
        'ethnic_group',
        'education_age',
        'working_at_baseline',
        'penetrating_injury'
    ]
    categorical_features = [
        'smoker',
        'iss_category'
        # iss_score?
        # others?
    ]
    """

    # === CYBERSECURITY ===
    dataset = DatasetName.CYBERSECURITY.value
    X = load_features(file_path="X_cybersecurity_intrusion_data.csv")
    y = load_labels(file_path="y_cybersecurity_intrusion_data.csv")

    nominal_features = [
        'encryption_used',
        'browser_type',
        'protocol_type'
    ]


    # === LUNG DISEASE ===
    """dataset = DatasetName.LUNG_DISEASE.value
    X = load_features(file_path="X_lung_disease_data.csv")
    y = load_labels(file_path="y_lung_disease_data.csv")
    
    nominal_features = [
        "Gender",
        "Smoking Status",
        "Disease Type",
        "Treatment Type"
    ]"""

    # TABLE DATA #

    # 1. logistic regression (no embedding), dataset: posttrauma
    (log_reg_dataset, log_reg_ml_method, log_reg_emb_method, log_reg_best_params,
     log_reg_pca_comp, log_reg_train_score, log_reg_test_scores) = \
        logistic_regression(dataset_name=dataset, X=X, y=y,  # test
                            nominal_features=nominal_features, pca=False)

    save_results_to_csv(
        dataset_name=log_reg_dataset,
        ml_method=log_reg_ml_method,
        emb_method="none",
        pca_n_comp=log_reg_pca_comp,
        best_params=log_reg_best_params,
        concatenation="no",
        is_train=True,
        metrics=log_reg_train_score,
        output_file=f"{dataset}_log_reg_train.csv")

    save_results_to_csv(
        dataset_name=log_reg_dataset,
        ml_method=log_reg_ml_method,
        emb_method="none",
        pca_n_comp=log_reg_pca_comp,
        best_params=log_reg_best_params,
        concatenation="no",
        is_train=False,
        metrics=log_reg_test_scores,
        output_file=f"{dataset}_log_reg_test.csv")

    # 2. log reg + random trees embedding
    (lr_rt_dataset, lr_rt_ml_method, lr_rt_emb_method, lr_rt_concatenation, lr_rte_best_params, lr_rte_train_score,
     log_reg_rt_emb_test_scores) = lr_rte(dataset_name=dataset, X=X, y=y, nominal_features=nominal_features, pca=False)
    save_results_to_csv(
        dataset_name=lr_rt_dataset,
        concatenation=lr_rt_concatenation,
        emb_method=lr_rt_emb_method,
        is_train=True,
        metrics=lr_rte_train_score,
        ml_method=lr_rt_ml_method,
        best_params=lr_rte_best_params,
        pca_n_comp="None",
        output_file=f"{dataset}_lr_rte_train.csv")

    save_results_to_csv(
        dataset_name=lr_rt_dataset,
        concatenation=lr_rt_concatenation,
        emb_method=lr_rt_emb_method,
        is_train=False,
        metrics=log_reg_rt_emb_test_scores,
        ml_method=lr_rt_ml_method,
        best_params=lr_rte_best_params,
        pca_n_comp="None",
        output_file=f"{dataset}_lr_rte_test.csv")

    # 3. hgbc (no embedding)
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

    # hgbc_rte(X=X_posttrauma, y=y_posttrauma, nominal_features=nominal_features)
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
    labels_local = [
        # f"Logistic Regression",
        # f"LogReg + RTE"
        # f"HGBC",
        # f"HGBC + RTE"
    ]
    train_scores_local = [
        # log_reg_train_score,
        # log_reg_rt_emb_train_score,
        # hgbc_train_score,
        # hgbc_rt_emb_train_score
    ]
    test_score_medians_local = [
        # np.median(log_reg_test_scores),
        # np.median(log_reg_rt_emb_test_scores),
        # np.median(hgbc_test_scores),
        # np.median(hgbc_rt_emb_test_scores)
    ]
    test_score_mins_local = [
        np.min(log_reg_test_scores),
        # np.min(log_reg_rt_emb_test_scores),
        # np.min(hgbc_test_scores),
        # np.min(hgbc_rt_emb_test_scores)
    ]
    test_score_maxs_local = [
        # np.max(log_reg_test_scores),
        # np.max(log_reg_rt_emb_test_scores),
        # np.max(hgbc_test_scores),
        # np.max(hgbc_rt_emb_test_scores)
    ]

    # Convert to arrays
    train_scores_local = np.array(train_scores_local)
    test_score_medians_local = np.array(test_score_medians_local)
    test_score_mins_local = np.array(test_score_mins_local)
    test_score_maxs_local = np.array(test_score_maxs_local)

            plot_bar_chart(
            filename=f"lr_and_rte_embeddings",
            labels=labels_local,
            train_scores=train_scores_local,
            test_score_medians=test_score_medians_local,
            test_score_mins=test_score_mins_local,
            test_score_maxs=test_score_maxs_local
    )"""
