import numpy as np
import pandas as pd

from csv_saver import save_results_to_csv
from helpers import (load_labels, load_features, load_summaries,
                     concat_lr_rte, concat_hgbc_rte, concat_txt_hgbc,
                     concat_lr_txt_emb)
from models import (feature_extractor_all_minilm_l6_v2, feature_extractor_ember_v1, feature_extractor_stella_en_400M_v5,
                    feature_extractor_sentence_t5_base, feature_extractor_gtr_t5_base)

"""from models import feature_extractor_medembed_small_v0_1, feature_extractor_medembed_base_v0_1, \
    feature_extractor_gte_small, feature_extractor_gte_base, feature_extractor_gte_base_en_v1_5, \
    feature_extractor_gist_small_embedding_v0, feature_extractor_gist_embedding_v0, feature_extractor_electra_small, \
    feature_extractor_electra_base, feature_extractor_simsce_sup, feature_extractor_simsce_unsup, \
    feature_extractor_e5_small_v2, feature_extractor_e5_base_v2, feature_extractor_bge_small_en_v1_5, \
    feature_extractor_bge_base_en_v1_5, feature_extractor_clinical, feature_extractor_bert"""

#from models import (feature_extractor_gte_large, feature_extractor_medembed_large_v0_1, feature_extractor_gte_large_en_v1_5)

from values import Dataset


def extract_features_to_csv(input_csv, nominal_features):
    df = pd.read_csv(input_csv)

    metrics = [col for col in df.columns if col not in nominal_features]
    nominals = [col for col in df.columns if col in nominal_features]

    df_metrics = df[metrics]
    df_nominals = df[nominals]

    df_metrics.to_csv("X_posttrauma_metrics", index=False)
    df_nominals.to_csv("X_posttrauma_nominals", index=False)


def run_text_concatenated():
    posttrauma_dataset = Dataset.POSTTRAUMA.value

    # load features and labels
    # Conc 1 Paket
    #all_summaries = load_summaries("Summaries.txt")
    #X_posttrauma_all = load_features(file_path="X.csv")

    # Conc 2 Paket
    # all_summaries = "Summaries.txt"
    # X_posttrauma_metrics = load_features(file_path="X_posttrauma_metrics.csv")

    # Conc 3 Paket
    nominal_summaries = "Summaries_nominal.txt"
    X_posttrauma_metrics = load_features(file_path="X_posttrauma_metrics.csv")
    conc_art = "_conc_3_"

    y_posttrauma = load_labels()

    nominal_features = [
        'gender_birth',
        'ethnic_group',
        'education_age',
        'working_at_baseline',
        'penetrating_injury'
    ]

    text_feature = 'text'

    categorical_features = [
        'smoker',
        'iss_category'
        # iss_score?
        # number of fractures?
        # others?
    ]

    feature_extractors = {

        # Stella en 400m v5
        "Stella-EN-400M-v5": feature_extractor_stella_en_400M_v5,

        # All MiniLM L6 v2
        "all_miniLM_L6_v2": feature_extractor_all_minilm_l6_v2,

        # GTR T5 Base
        "GTR_T5_Base": feature_extractor_gtr_t5_base,

        # Sentence T5 Base
        "sentence_t5_base": feature_extractor_sentence_t5_base,

        # modernbert-embed-base
        #"modernbert_embed_base": feature_extractor_mbert_embed_base,

        # GTE modernbert base
        #"gte_modernbert_base": feature_extractor_gte_mbert_base,

        # Ember v1
        "ember_v1": feature_extractor_ember_v1

        # Clinical Longformer
        #"Clinical-Longformer": feature_extractor_clinical,

        # BERT (half done)
        #"BERT-Base": feature_extractor_bert,

        # ELECTRA (half done)
        #"ELECTRA-Small": feature_extractor_electra_small,
        #"ELECTRA-Base": feature_extractor_electra_base,
        #"ELECTRA-Large": feature_extractor_electra_large,

        # SimSCE (done)
        #"SimSCE-Sup": feature_extractor_simsce_sup,
        #"SimSCE-Unsup": feature_extractor_simsce_unsup,

        # E5 Models
        #"E5-Small-V2": feature_extractor_e5_small_v2,
        #"E5-Base-V2": feature_extractor_e5_base_v2,
        #"E5-Large-V2": feature_extractor_e5_large_v2,

        # BGE Models (done)
        #"BGE-Small-EN-v1.5": feature_extractor_bge_small_en_v1_5,
        #"BGE-Base-EN-v1.5": feature_extractor_bge_base_en_v1_5,
        #"BGE-Large-EN-v1.5": feature_extractor_bge_large_en_v1_5,

        # GIST Models
        #"GIST-Small-Embedding-v0": feature_extractor_gist_small_embedding_v0,
        #"GIST-Embedding-v0": feature_extractor_gist_embedding_v0,
        #"GIST-Large-Embedding-v0": feature_extractor_gist_large_embedding_v0,

        # MedEmbed Models (problem)
        #"MedEmbed-Small-v0.1": feature_extractor_medembed_small_v0_1,  # (problem)"""
        # "MedEmbed-Base-v0.1": feature_extractor_medembed_base_v0_1,  # (problem)
        #
        #"MedEmbed-Large-v0.1": feature_extractor_medembed_large_v0_1, # (problem)

        # Potion Models
        # "Potion-Base-2M": feature_extractor_potion_base_2M,
        # "Potion-Base-4M": feature_extractor_potion_base_4M,
        # "Potion-Base-8M": feature_extractor_potion_base_8M,

        # GTE Models
        #"GTE-Small": feature_extractor_gte_small,  # (done)
        #"GTE-Base": feature_extractor_gte_base,  # (done)
        #"GTE-Base-EN-v1.5": feature_extractor_gte_base_en_v1_5,  #(ready)
        #"GTE-Large": feature_extractor_gte_large,  # (done)
        #"GTE-Large-EN-v1.5": feature_extractor_gte_large_en_v1_5, # (ready)

        # Stella Model
        # "Stella-EN-400M-v5": feature_extractor_stella_en_400M_v5 # (not ready)
    }

    # Calculate results for each model
    for model_name, feature_extractor in feature_extractors.items():
        print("Begin conc methods")
        # HGBC TXT Concatenation
        (hgbc_conc_dataset, hgbc_conc_ml_method, hgbc_conc_emb_method,
         hgbc_conc_yesno, hgbc_best_params, hgbc_pca_components, hgbc_conc_train_score,
         hgbc_conc_test_scores) = concat_txt_hgbc(
            dataset_name=posttrauma_dataset,
            emb_method=model_name,
            feature_extractor=feature_extractor,
            raw_text_summaries=nominal_summaries,
            X_tabular=X_posttrauma_metrics, y=y_posttrauma,
            nominal_features=nominal_features,
            text_feature_column_name=text_feature,
            n_repeats=10,
            #n_repeats=1,
            n_components=None)

        save_results_to_csv(output_file=f"{model_name}_HGBC{conc_art}train_all_sum_all_metrics.csv",
                            dataset_name=hgbc_conc_dataset,
                            ml_method=hgbc_conc_ml_method,
                            emb_method=hgbc_conc_emb_method,
                            concatenation=hgbc_conc_yesno,
                            best_params=hgbc_best_params,
                            pca_n_comp=hgbc_pca_components,
                            metrics=hgbc_conc_train_score,
                            is_train=True)

        save_results_to_csv(output_file=f"{model_name}_HGBC{conc_art}test_all_sum_all_metrics.csv",
                            dataset_name=hgbc_conc_dataset,
                            ml_method=hgbc_conc_ml_method,
                            emb_method=hgbc_conc_emb_method,
                            concatenation=hgbc_conc_yesno,
                            best_params=hgbc_best_params,
                            pca_n_comp=hgbc_pca_components,
                            metrics=hgbc_conc_test_scores,
                            is_train=False)

        (lr_conc_dataset, lr_conc_ml_method, lr_conc_emb_method,
         lr_conc_yesno, lr_best_params, lr_pca_components, lr_conc_train_score,
         lr_conc_test_scores) = concat_lr_txt_emb(
            dataset_name=posttrauma_dataset,
            emb_method=model_name,
            feature_extractor=feature_extractor,
            raw_text_summaries=nominal_summaries,
            X_tabular=X_posttrauma_metrics, y=y_posttrauma,
            nominal_features=nominal_features,
            text_feature_column_name=text_feature,
            imp_max_iter=30, class_max_iter=10000,
            n_components=None, n_repeats=10)

        save_results_to_csv(output_file=f"{model_name}_LR{conc_art}train_all_sum_all_metrics.csv",
                            dataset_name=lr_conc_dataset,
                            ml_method=lr_conc_ml_method,
                            emb_method=lr_conc_emb_method,
                            concatenation=lr_conc_yesno,
                            best_params=lr_best_params,
                            pca_n_comp=lr_pca_components,
                            metrics=lr_conc_train_score,
                            is_train=True)

        save_results_to_csv(output_file=f"{model_name}_LR{conc_art}test_all_sum_all_metrics.csv",
                            dataset_name=lr_conc_dataset,
                            ml_method=lr_conc_ml_method,
                            emb_method=lr_conc_emb_method,
                            concatenation=lr_conc_yesno,
                            best_params=lr_best_params,
                            pca_n_comp=lr_pca_components,
                            metrics=lr_conc_test_scores,
                            is_train=False)

        """
        # Bar chart geht gerade nicht, da train/test scores jetzt verschiedene Metriken enthalten
        labels_local = [
            f"{model_name} \n+ LogReg conc.",
            #f"{model_name} \n+ HGBC conc."
        ]
        train_scores_local = [
            lr_conc_train_score,
            #hgbc_conc_train_score
        ]
        test_score_medians_local = [
            np.median(lr_conc_test_scores),
            #np.median(hgbc_conc_test_scores)
        ]
        test_score_mins_local = [
            np.min(lr_conc_test_scores),
            #np.min(hgbc_conc_test_scores)
        ]
        test_score_maxs_local = [
            np.max(lr_conc_test_scores),
            #np.max(hgbc_conc_test_scores)
        ]

        # Convert to arrays
        train_scores_local = np.array(train_scores_local)
        test_score_medians_local = np.array(test_score_medians_local)
        test_score_mins_local = np.array(test_score_mins_local)
        test_score_maxs_local = np.array(test_score_maxs_local)

        plot_bar_chart(
            filename=f"{model_name}_conc_02_02",
            labels=labels_local,
            train_scores=train_scores_local,
            test_score_medians=test_score_medians_local,
            test_score_mins=test_score_mins_local,
            test_score_maxs=test_score_maxs_local
        )"""


def run_rte_concatenated():
    posttrauma_dataset = Dataset.POSTTRAUMA.value

    # load features and labels
    X_posttrauma = load_features()
    y_posttrauma = load_labels()

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

    (lr_rt_conc_dataset, lr_rt_conc_ml_method, lr_rt_conc_emb_method,
     lr_rt_conc_yesno, lr_rt_best_params, lr_rt_pca_components,
     lr_rt_conc_train_score, lr_rt_conc_test_scores) = concat_lr_rte(
        dataset_name=posttrauma_dataset,
        X_tabular=X_posttrauma, y=y_posttrauma,
        nominal_features=nominal_features, pca_n_comp=None,
        n_repeats=10, class_max_iter=10000, imp_max_iter=50)

    # todo:save train&test results as list and iterate
    save_results_to_csv(output_file=f"LR_rte_conc_train.csv",
                        dataset_name=lr_rt_conc_dataset,
                        ml_method=lr_rt_conc_ml_method,
                        emb_method=lr_rt_conc_emb_method,
                        concatenation=lr_rt_conc_yesno,
                        best_params=lr_rt_best_params,
                        pca_n_comp=lr_rt_pca_components,
                        metrics=lr_rt_conc_train_score,
                        is_train=True)

    save_results_to_csv(output_file=f"LR_rte_conc_test.csv",
                        dataset_name=lr_rt_conc_dataset,
                        ml_method=lr_rt_conc_ml_method,
                        emb_method=lr_rt_conc_emb_method,
                        concatenation=lr_rt_conc_yesno,
                        best_params=lr_rt_best_params,
                        pca_n_comp=lr_rt_pca_components,
                        metrics=lr_rt_conc_test_scores,
                        is_train=False)

    # HGBC Concatenated (Tab. + RT Embeddings)
    (hgbc_conc_rte_dataset, hgbc_conc_rte_ml_method, hgbc_conc_rte_emb_method,
     hgbc_rte_conc, hgbc_rte_best_params, hgbc_rte_pca_n_comp,
     hgbc_conc_rte_train_score, hgbc_conc_rte_test_scores) = concat_hgbc_rte(dataset_name=posttrauma_dataset,
                                                                             X_tabular=X_posttrauma,
                                                                             nominal_features=nominal_features,
                                                                             n_repeats=10,
                                                                             y=y_posttrauma,
                                                                             imp_max_iter=50,
                                                                             pca_n_comp=None)

    save_results_to_csv(output_file=f"HGBC_conc_rte_train.csv", dataset_name=hgbc_conc_rte_dataset,
                        ml_method=hgbc_conc_rte_ml_method, emb_method=hgbc_conc_rte_emb_method,
                        concatenation=hgbc_rte_conc, metrics=hgbc_conc_rte_train_score,
                        best_params=hgbc_rte_best_params, pca_n_comp=hgbc_rte_pca_n_comp, is_train=True)

    save_results_to_csv(output_file=f"HGBC_conc_rte_test.csv", dataset_name=hgbc_conc_rte_dataset,
                        ml_method=hgbc_conc_rte_ml_method, emb_method=hgbc_conc_rte_emb_method,
                        concatenation=hgbc_rte_conc, metrics=hgbc_conc_rte_test_scores,
                        best_params=hgbc_rte_best_params, pca_n_comp=hgbc_rte_pca_n_comp, is_train=False)
    """
    labels_local = [
        f"LogReg Table + RTE",
        f"HGBC Table + RTE."
        ]
    train_scores_local = [
        lr_rt_conc_train_score,
        hgbc_conc_rte_train_score
        ]
    test_score_medians_local = [
        np.median(lr_rt_conc_test_scores),
        np.median(hgbc_conc_rte_test_scores)
        ]
    test_score_mins_local = [
        np.min(lr_rt_conc_test_scores),
        np.min(hgbc_conc_rte_test_scores)
        ]
    test_score_maxs_local = [
        np.max(lr_rt_conc_test_scores),
        np.max(hgbc_conc_rte_test_scores)
    ]

    # Convert to arrays
    train_scores_local = np.array(train_scores_local)
    test_score_medians_local = np.array(test_score_medians_local)
    test_score_mins_local = np.array(test_score_mins_local)
    test_score_maxs_local = np.array(test_score_maxs_local)

    plot_bar_chart(
        filename=f"RTE_conc",
        labels=labels_local,
        train_scores=train_scores_local,
        test_score_medians=test_score_medians_local,
        test_score_mins=test_score_mins_local,
        test_score_maxs=test_score_maxs_local
    )
    """
