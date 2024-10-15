import numpy as np

from helpers import load_labels, load_features, load_summaries, lg_reg, lg_reg_rt_emb, lg_reg_txt_emb, \
    embedding_cls, embedding_mean_with_cls_and_sep, embedding_mean_without_cls_and_sep
from bar_plotting import plot_bar_chart


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

    # load patient summaries for embeddings
    patient_summaries = load_summaries()

    # no embedding
    lg_reg_train_score, lg_reg_test_scores = \
        lg_reg(X=X, y=y, nominal_features=nominal_features)
    # random trees embedding
    lg_reg_rt_emb_train_score, lg_reg_rt_emb_test_scores = \
        lg_reg_rt_emb(X=X, y=y, nominal_features=nominal_features)
    # embedding based on [CLS] token
    lg_reg_emb_cls_train_score, lg_reg_emb_cls_test_scores = \
        lg_reg_txt_emb(embeddings=embedding_cls(patient_summaries), y=y)
    # mean embedding including [CLS] and [SEP] tokens
    lg_reg_emb_mean_with_cls_and_sep_train_score, lg_reg_emb_mean_with_cls_and_sep_test_scores = \
        lg_reg_txt_emb(embeddings=embedding_mean_with_cls_and_sep(patient_summaries), y=y)
    # mean embedding excluding [CLS] and [SEP] tokens
    lg_reg_emb_mean_without_cls_and_sep_train_score, lg_reg_emb_mean_without_cls_and_sep_test_scores = \
        lg_reg_txt_emb(embeddings=embedding_mean_without_cls_and_sep(patient_summaries), y=y)
    
    labels = [
        'No embedding',
        'Random trees\nembedding',
        'Embedding based\non [CLS] token',
        'Mean embedding\nincluding [CLS]\nand [SEP] tokens',
        'Mean embedding\nexcluding [CLS]\nand [SEP] tokens'
    ]
    train_scores = np.array([
        lg_reg_train_score,
        lg_reg_rt_emb_train_score,
        lg_reg_emb_cls_train_score,
        lg_reg_emb_mean_with_cls_and_sep_train_score,
        lg_reg_emb_mean_without_cls_and_sep_train_score
    ])
    test_score_medians = np.array([
        np.median(lg_reg_test_scores),
        np.median(lg_reg_rt_emb_test_scores),
        np.median(lg_reg_emb_cls_test_scores),
        np.median(lg_reg_emb_mean_with_cls_and_sep_test_scores),
        np.median(lg_reg_emb_mean_without_cls_and_sep_test_scores)
    ])
    test_score_mins = np.array([
        np.min(lg_reg_test_scores),
        np.min(lg_reg_rt_emb_test_scores),
        np.min(lg_reg_emb_cls_test_scores),
        np.min(lg_reg_emb_mean_with_cls_and_sep_test_scores),
        np.min(lg_reg_emb_mean_without_cls_and_sep_test_scores)
    ])
    test_score_maxs = np.array([
        np.max(lg_reg_test_scores),
        np.max(lg_reg_rt_emb_test_scores),
        np.max(lg_reg_emb_cls_test_scores),
        np.max(lg_reg_emb_mean_with_cls_and_sep_test_scores),
        np.max(lg_reg_emb_mean_without_cls_and_sep_test_scores)
    ])
    
    # plot the bars
    plot_bar_chart(labels, train_scores, test_score_medians, test_score_mins, test_score_maxs)
