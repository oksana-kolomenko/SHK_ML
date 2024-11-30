import numpy as np
from helpers import (load_labels, load_features, load_summaries, logistic_regression,
                     lr_ran_tree_emb, lr_txt_emb, hgbc, hgbc_txt_emb)
from bar_plotting import plot_bar_chart
from models import feature_extractor_clinical


def run_all_models():

    # load features and labels
    X = load_features()
    y = load_labels()

    # define nominal features
    nominal_features = [
        'gender_birth',
        'ethnic_group',
        'education_age',
        'working_at_baseline',
        'penetrating_injury'
    ]

    cl_feature_extractor = feature_extractor_clinical

    # load patient summaries for embeddings
    patient_summaries = load_summaries()

    # 1. logistic regression (no embedding)
    lg_reg_train_score, lg_reg_test_scores = \
        logistic_regression(X=X, y=y, nominal_features=nominal_features)

    # 2. random trees embedding
    lg_reg_rt_emb_train_score, lg_reg_rt_emb_test_scores = \
        lr_ran_tree_emb(X=X, y=y, nominal_features=nominal_features)

    # 3. cl text embedding
    text_emb_cls_train_score, text_emb_cls_test_scores = \
        lr_txt_emb(feature_extractor=cl_feature_extractor, summaries=patient_summaries, y=y)

    # 4. hgbc (no embedding)
    hgbc_train_score, hgbc_test_scores = \
        hgbc(X=X, y=y, nominal_features=nominal_features)

    # 5. random trees embedding + hgbc
    hgbc_rt_emb_train_score, hgbc_rt_emb_test_scores = \
        lr_ran_tree_emb(X=X, y=y, nominal_features=nominal_features)

    # 6. cl text embedding + hgbc
    text_emb_hgbc_cls_train_score, text_emb_hgbc_cls_test_scores = \
        hgbc_txt_emb(feature_extractor=cl_feature_extractor,
                     summaries=patient_summaries, nominal_features=nominal_features, y=y)

    labels = [
        'Logistic Regression',
        'Random trees\nembedding + Logistic Regression',
        'Text Embedding \n+ Logistic Regression',
        'No embedding + hgbc',
        'Random trees\nembedding + hgbc',
        'Text Embedding + hgbc'
    ]
    train_scores = np.array([
        lg_reg_train_score,
        lg_reg_rt_emb_train_score,
        text_emb_cls_train_score,
        hgbc_train_score,
        hgbc_rt_emb_train_score,
        text_emb_hgbc_cls_train_score
    ])
    test_score_medians = np.array([
        np.median(lg_reg_test_scores),
        np.median(lg_reg_rt_emb_test_scores),
        np.median(text_emb_cls_test_scores),
        np.median(hgbc_test_scores),
        np.median(hgbc_rt_emb_test_scores),
        np.median(text_emb_hgbc_cls_test_scores)
    ])
    test_score_mins = np.array([
        np.min(lg_reg_test_scores),
        np.min(lg_reg_rt_emb_test_scores),
        np.min(text_emb_cls_test_scores),
        np.min(hgbc_test_scores),
        np.min(hgbc_rt_emb_test_scores),
        np.min(text_emb_hgbc_cls_test_scores)
    ])
    test_score_maxs = np.array([
        np.max(lg_reg_test_scores),
        np.max(lg_reg_rt_emb_test_scores),
        np.max(text_emb_cls_test_scores),
        np.max(hgbc_test_scores),
        np.max(hgbc_rt_emb_test_scores),
        np.max(text_emb_hgbc_cls_test_scores)
    ])

    # plot the bars
    plot_bar_chart(filename="bar_chart_txt", labels=labels, train_scores=train_scores,
                   test_score_medians=test_score_medians, test_score_mins=test_score_mins,
                   test_score_maxs=test_score_maxs)
