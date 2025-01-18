import numpy as np

from csv_saver import save_results_to_csv
from helpers import (load_labels, load_features, load_summaries, logistic_regression,
                     lr_rt_emb, hgbc, hgbc_ran_tree_emb)
from bar_plotting import plot_bar_chart
from values import Dataset


def run_models_on_table_data():
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

    categorical_features = [
        'smoker',
        'iss_category'
        # iss_score?
        # others?
    ]

    # TABLE DATA #

    """  # 1. logistic regression (no embedding), dataset: posttrauma
    log_reg_dataset, log_reg_ml_method, log_reg_emb_method, log_reg_train_score, log_reg_test_scores = \
        logistic_regression(dataset_name=posttrauma_dataset, X=X_posttrauma, y=y_posttrauma,  # test
                            nominal_features=nominal_features)

    save_results_to_csv( # anpassen
    dataset_name=log_reg_dataset,
    ml_method=log_reg_ml_method,
    output_file="no_emb_log_reg_train.csv")

    # test ...
    """

    # 2. log reg + random trees embedding
    (lr_rt_dataset, lr_rt_ml_method, lr_rt_emb_method, lr_rt_concatenation, log_reg_rt_emb_train_score,
     log_reg_rt_emb_test_scores) = lr_rt_emb(dataset_name=posttrauma_dataset, X=X_posttrauma, y=y_posttrauma,
                                             nominal_features=nominal_features)
    save_results_to_csv(
        dataset_name=lr_rt_dataset,
        concatenation=lr_rt_concatenation,
        emb_method=lr_rt_emb_method,
        is_train=True,
        metrics=log_reg_rt_emb_train_score,
        ml_method=lr_rt_ml_method,
        output_file="rte_log_reg_train.csv")

    save_results_to_csv(
        dataset_name=lr_rt_dataset,
        concatenation=lr_rt_concatenation,
        emb_method=lr_rt_emb_method,
        is_train=False,
        metrics=log_reg_rt_emb_test_scores,
        ml_method=lr_rt_ml_method,
        output_file="rte_log_reg_test.csv")

    # test ...

    """
    # 3. hgbc (no embedding)
    hgbc_train_score, hgbc_test_scores = \
        hgbc(dataset_name=posttrauma_dataset, X=X_posttrauma, y=y_posttrauma, nominal_features=nominal_features)

    # 4. random trees embedding + hgbc
    hgbc_rt_emb_train_score, hgbc_rt_emb_test_scores = \
        lr_ran_tree_emb(dataset_name=posttrauma_dataset, X=X_posttrauma, y=y_posttrauma, 
                        nominal_features=nominal_features)
    """
    labels_local = [
           # f"Logistic Regression",
            f"LogReg + RTE"
            #f"HGBC",
            #f"HGBC + RTE"
        ]
    train_scores_local = [
            #log_reg_train_score,
            log_reg_rt_emb_train_score,
            #hgbc_train_score,
            #hgbc_rt_emb_train_score
        ]
    test_score_medians_local = [
            #np.median(log_reg_test_scores),
            np.median(log_reg_rt_emb_test_scores),
            #np.median(hgbc_test_scores),
            #np.median(hgbc_rt_emb_test_scores)
        ]
    test_score_mins_local = [
            #np.min(log_reg_test_scores),
            np.min(log_reg_rt_emb_test_scores),
            #np.min(hgbc_test_scores),
            #np.min(hgbc_rt_emb_test_scores)
        ]
    test_score_maxs_local = [
            #np.max(log_reg_test_scores),
            np.max(log_reg_rt_emb_test_scores),
            #np.max(hgbc_test_scores),
            #np.max(hgbc_rt_emb_test_scores)
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
    )
