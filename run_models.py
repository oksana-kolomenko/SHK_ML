import numpy as np
from new_file import (load_labels, load_features, load_summaries, lg_reg, lg_reg_emb,
                      lg_re_txt_emb, embedding_with_cls_token, embedding_with_cls_and_sep_tokens,
                      embedding_without_cls_and_sep_tokens, lg_reg_emb_new_pp, lg_reg_new_pp)
from bar_plotting import plot_bar_chart


def run_all_models():
    X = load_features()
    y = load_labels()
    patient_summaries = load_summaries()
    #emb_with_cls_token = embedding_with_cls_token(patient_summaries)
    #emb_with_cls_and_sep_tokens = embedding_with_cls_and_sep_tokens(patient_summaries)
    #emb_without_cls_and_sep_tokens = embedding_without_cls_and_sep_tokens(patient_summaries)

    #lr_model_train_score, lr_model_test_scores =
    lg_reg(X=X, y=y)
    lg_reg_new_pp(X=X, y=y)
    # rt_model_train_score, rt_model_test_scores = (
    # lg_reg_emb_new_pp(X=X, y=y)
    # lg_reg_emb(X=X, y=y)

    #lr_txt_model_train_score_1, lr_txt_model_test_scores_1 =
    #lg_re_txt_emb(embeddings=emb_with_cls_token, y=y)
    #lr_txt_model_train_score_2, lr_txt_model_test_scores_2 = lg_re_txt_emb(embeddings=emb_with_cls_and_sep_tokens, y=y)
    #lr_txt_model_train_score_3, lr_txt_model_test_scores_3 = lg_re_txt_emb(embeddings=emb_without_cls_and_sep_tokens, y=y)

    # plot the bars
    """labels = ['With embedding', 'Embedding with cls token', 'Embedding with cls and sep tokens',
              'Embedding without cls and sep tokens', 'Without embedding']
    train_scores = [rt_model_train_score, lr_txt_model_train_score_1, lr_txt_model_train_score_2,
                    lr_txt_model_train_score_3, lr_model_train_score]
    test_means = [np.mean(rt_model_test_scores), np.mean(lr_txt_model_test_scores_1),
                  np.mean(lr_txt_model_test_scores_2), np.mean(lr_txt_model_test_scores_3), np.mean(lr_model_test_scores)]
    test_stds = [np.std(rt_model_test_scores), np.std(lr_txt_model_test_scores_1), np.std(lr_txt_model_test_scores_2),
                 np.std(lr_txt_model_test_scores_3), np.std(lr_model_test_scores)]

    plot_bar_chart(labels, train_scores, test_means, test_stds)"""
