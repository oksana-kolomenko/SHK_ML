import numpy as np
from helpers import load_labels, load_features, load_summaries, lg_reg, lg_reg_rt_emb, lg_reg_txt_emb, \
    embedding_cls, embedding_mean_with_cls_and_sep, embedding_mean_without_cls_and_sep, lg_reg_hgbc, \
    lg_reg_hgbc_txt_emb, lr_agg_txt_emb, lr_agg_hgbc_txt_emb
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

    # no embedding + hgbc
    lg_reg_hgbc_train_score, lg_reg_hgbc_test_scores = \
        lg_reg_hgbc(X=X, y=y, nominal_features=nominal_features)

    # random trees embedding
    lg_reg_rt_emb_train_score, lg_reg_rt_emb_test_scores = \
        lg_reg_rt_emb(X=X, y=y, nominal_features=nominal_features)

    # random trees embedding + hgbc
    lg_reg_hgbc_rt_emb_train_score, lg_reg_hgbc_rt_emb_test_scores = \
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

    # embedding based on [CLS] token + hgbc
    lg_reg_emb_hgbc_cls_train_score, lg_reg_emb_hgbc_cls_test_scores = \
        lg_reg_hgbc_txt_emb(embeddings=embedding_cls(patient_summaries), nominal_features=nominal_features, y=y)
    # mean embedding including [CLS] and [SEP] tokens
    lg_reg_emb_hgbc_mean_with_cls_and_sep_train_score, lg_reg_emb_hgbc_mean_with_cls_and_sep_test_scores = \
        lg_reg_hgbc_txt_emb(embeddings=embedding_mean_with_cls_and_sep(patient_summaries),
                            nominal_features=nominal_features, y=y)
    # mean embedding excluding [CLS] and [SEP] tokens
    lg_reg_emb_hgbc_mean_without_cls_and_sep_train_score, lg_reg_emb_hgbc_mean_without_cls_and_sep_test_scores = \
        lg_reg_hgbc_txt_emb(embeddings=embedding_mean_without_cls_and_sep(patient_summaries),
                            nominal_features=nominal_features, y=y)

    # embedding based on [CLS] token + aggregation
    lr_txt_emb_agg_cls_train_score, lr_txt_emb_agg_cls_test_scores = \
        lr_agg_txt_emb(embeddings=embedding_cls(patient_summaries), y=y)
    # mean embedding including [CLS] and [SEP] tokens
    lr_txt_emb_agg_mean_with_cls_and_sep_train_score, lr_txt_emb_agg_mean_with_cls_and_sep_test_scores = \
        lr_agg_txt_emb(embeddings=embedding_mean_with_cls_and_sep(patient_summaries), y=y)
    # mean embedding excluding [CLS] and [SEP] tokens
    lr_txt_emb_agg_mean_without_cls_and_sep_train_score, lr_txt_emb_agg_mean_without_cls_and_sep_test_scores = \
        lr_agg_txt_emb(embeddings=embedding_mean_without_cls_and_sep(patient_summaries), y=y)

    # embedding based on [CLS] token + aggregation + hgbc
    lr_txt_emb_agg_hgbc_cls_train_score, lr_txt_emb_agg_hgbc_cls_test_scores = \
        lr_agg_hgbc_txt_emb(embeddings=embedding_cls(patient_summaries),nominal_features=nominal_features, y=y)
    # mean embedding including [CLS] and [SEP] tokens
    lr_txt_emb_agg_hgbc_mean_with_cls_and_sep_train_score, lr_txt_emb_agg_hgbc_mean_with_cls_and_sep_test_scores = \
        lr_agg_hgbc_txt_emb(embeddings=embedding_mean_with_cls_and_sep(patient_summaries),
                            nominal_features=nominal_features, y=y)
    # mean embedding excluding [CLS] and [SEP] tokens
    lr_txt_emb_agg_hgbc_mean_without_cls_and_sep_train_score, lr_txt_emb_agg_hgbc_mean_without_cls_and_sep_test_scores = \
        lr_agg_hgbc_txt_emb(embeddings=embedding_mean_without_cls_and_sep(patient_summaries),
                            nominal_features=nominal_features, y=y)

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


    hgbc_labels = [
        'No embedding + hgbc',
        'Random trees\nembedding + hgbc',
        'Embedding based\non [CLS] token + hgbc',
        'Mean embedding\nincluding [CLS]\nand [SEP] tokens + hgbc',
        'Mean embedding\nexcluding [CLS]\nand [SEP] tokens + hgbc'
    ]
    hgbc_train_scores = np.array([
        lg_reg_hgbc_train_score,
        lg_reg_hgbc_rt_emb_train_score,
        lg_reg_emb_hgbc_cls_train_score,
        lg_reg_emb_hgbc_mean_with_cls_and_sep_train_score,
        lg_reg_emb_hgbc_mean_without_cls_and_sep_train_score
    ])
    hgbc_test_score_medians = np.array([
        np.median(lg_reg_hgbc_test_scores),
        np.median(lg_reg_hgbc_rt_emb_test_scores),
        np.median(lg_reg_emb_hgbc_cls_test_scores),
        np.median(lg_reg_emb_hgbc_mean_with_cls_and_sep_test_scores),
        np.median(lg_reg_emb_hgbc_mean_without_cls_and_sep_test_scores)
    ])
    hgbc_test_score_mins = np.array([
        np.min(lg_reg_hgbc_test_scores),
        np.min(lg_reg_hgbc_rt_emb_test_scores),
        np.min(lg_reg_emb_hgbc_cls_test_scores),
        np.min(lg_reg_emb_hgbc_mean_with_cls_and_sep_test_scores),
        np.min(lg_reg_emb_hgbc_mean_without_cls_and_sep_test_scores)
    ])
    hgbc_test_score_maxs = np.array([
        np.max(lg_reg_hgbc_test_scores),
        np.max(lg_reg_hgbc_rt_emb_test_scores),
        np.max(lg_reg_emb_hgbc_cls_test_scores),
        np.max(lg_reg_emb_hgbc_mean_with_cls_and_sep_test_scores),
        np.max(lg_reg_emb_hgbc_mean_without_cls_and_sep_test_scores),
    ])

    agg_labels = [
        'Embedding based\non [CLS] token',
        'Mean embedding\nincluding [CLS]\nand [SEP] tokens',
        'Mean embedding\nexcluding [CLS]\nand [SEP] tokens',
        'Embedding based\non [CLS] token + hgbc',
        'Mean embedding\nincluding [CLS]\nand [SEP] tokens + hgbc',
        'Mean embedding\nexcluding [CLS]\nand [SEP] tokens + hgbc'
    ]
    agg_train_scores = np.array([
        lr_txt_emb_agg_cls_train_score,
        lr_txt_emb_agg_mean_with_cls_and_sep_train_score,
        lr_txt_emb_agg_mean_without_cls_and_sep_train_score,
        lr_txt_emb_agg_hgbc_cls_train_score,
        lr_txt_emb_agg_hgbc_mean_with_cls_and_sep_train_score,
        lr_txt_emb_agg_hgbc_mean_without_cls_and_sep_train_score
    ])
    agg_test_score_medians = np.array([
        np.median(lr_txt_emb_agg_cls_test_scores),
        np.median(lr_txt_emb_agg_mean_with_cls_and_sep_test_scores),
        np.median(lr_txt_emb_agg_mean_without_cls_and_sep_test_scores),
        # with hgbc
        np.median(lr_txt_emb_agg_hgbc_cls_train_score),
        np.median(lr_txt_emb_agg_hgbc_mean_with_cls_and_sep_test_scores),
        np.median(lr_txt_emb_agg_hgbc_mean_without_cls_and_sep_test_scores),
    ])
    agg_test_score_mins = np.array([
        np.min(lr_txt_emb_agg_cls_test_scores),
        np.min(lr_txt_emb_agg_mean_with_cls_and_sep_test_scores),
        np.min(lr_txt_emb_agg_mean_without_cls_and_sep_test_scores),
        # with hgbc
        np.min(lr_txt_emb_agg_hgbc_cls_train_score),
        np.min(lr_txt_emb_agg_hgbc_mean_with_cls_and_sep_test_scores),
        np.min(lr_txt_emb_agg_hgbc_mean_without_cls_and_sep_test_scores),
    ])
    agg_test_score_maxs = np.array([
        np.max(lr_txt_emb_agg_cls_test_scores),
        np.max(lr_txt_emb_agg_mean_with_cls_and_sep_test_scores),
        np.max(lr_txt_emb_agg_mean_without_cls_and_sep_test_scores),
        # with hgbc
        np.max(lr_txt_emb_agg_hgbc_cls_train_score),
        np.max(lr_txt_emb_agg_hgbc_mean_with_cls_and_sep_test_scores),
        np.max(lr_txt_emb_agg_hgbc_mean_without_cls_and_sep_test_scores),
    ])
    
    # plot the bars
    #plot_bar_chart(filename="bar_chart", labels=labels, train_scores=train_scores,
     #              test_score_medians=test_score_medians, test_score_mins=test_score_mins,
      #             test_score_maxs=test_score_maxs)

    # plot the bars with hgbc
    #plot_bar_chart(filename="bar_chart_hgbc", labels=hgbc_labels, train_scores=hgbc_train_scores,
     #              test_score_medians=hgbc_test_score_medians, test_score_mins=hgbc_test_score_mins,
      #             test_score_maxs=hgbc_test_score_maxs)

    # plot the bars with aggregation
    plot_bar_chart(filename="bar_chart_agg", labels=agg_labels, train_scores=agg_train_scores,
                   test_score_medians=agg_test_score_medians, test_score_mins=agg_test_score_mins,
                   test_score_maxs=agg_test_score_maxs)
