import numpy as np

from csv_saver import save_results_to_csv
from helpers import (load_labels, load_features, load_summaries, concat_lr_txt_emb)
from bar_plotting import plot_bar_chart
from models import feature_extractor_clinical
from values import Dataset


def run_models_concatenated():
    posttrauma_dataset = Dataset.POSTTRAUMA.value

    # load features and labels
    patient_summaries = load_summaries()
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
    feature_extractors = {
        # Clinical Longformer
        "Clinical-Longformer": feature_extractor_clinical,

        # BERT (half done)
        # "BERT": feature_extractor_bert,

        # ELECTRA (half done)
        # "ELECTRA-Small": feature_extractor_electra_small,
        # "ELECTRA-Base": feature_extractor_electra_base,
        # "ELECTRA-Large": feature_extractor_electra_large,

        # SimSCE (done)
        # "SimSCE-Sup": feature_extractor_simsce_sup,
        # "SimSCE-Unsup": feature_extractor_simsce_unsup,

        # E5 Models
        # "E5-Small-V2": feature_extractor_e5_small_v2,
        # "E5-Base-V2": feature_extractor_e5_base_v2,
        # "E5-Large-V2": feature_extractor_e5_large_v2,

        # BGE Models (done)
        # "BGE-Small-EN-v1.5": feature_extractor_bge_small_en_v1_5,
        # "BGE-Base-EN-v1.5": feature_extractor_bge_base_en_v1_5,
        # "BGE-Large-EN-v1.5": feature_extractor_bge_large_en_v1_5,

        # GIST Models
        # "GIST-Small-Embedding-v0": feature_extractor_gist_small_embedding_v0,
        # "GIST-Embedding-v0": feature_extractor_gist_embedding_v0,
        # "GIST-Large-Embedding-v0": feature_extractor_gist_large_embedding_v0,

        # MedEmbed Models (problem)
        # "MedEmbed-Small-v0.1": feature_extractor_medembed_small_v0_1, # (problem)
        # "MedEmbed-Base-v0.1": feature_extractor_medembed_base_v0_1, # (problem)
        # "MedEmbed-Large-v0.1": feature_extractor_medembed_large_v0_1, # (problem)

        # Potion Models
        # "Potion-Base-2M": feature_extractor_potion_base_2M,
        # "Potion-Base-4M": feature_extractor_potion_base_4M,
        # "Potion-Base-8M": feature_extractor_potion_base_8M,

        # GTE Models
        # "GTE-Small": feature_extractor_gte_small,  # (done)
        # "GTE-Base": feature_extractor_gte_base,  # (done)
        # "GTE-Base-EN-v1.5": feature_extractor_gte_base_en_v1_5, #(ready)
        # "GTE-Large": feature_extractor_gte_large,  # (done)
        # "GTE-Large-EN-v1.5": feature_extractor_gte_large_en_v1_5, # (ready)

        # Stella Model
        # "Stella-EN-400M-v5": feature_extractor_stella_en_400M_v5 # (not ready)
    }

    # TEXT EMBEDDINGS #

    # Calculate results for each model
    for model_name, feature_extractor in feature_extractors.items():
        # Log. Reg. Concatenated (Tab. + Text Embeddings)
        (lr_conc_dataset, lr_conc_ml_method, lr_conc_emb_method,
         lr_conc_train_score, lr_conc_test_scores) = concat_lr_txt_emb(dataset_name=posttrauma_dataset,
                                                                       emb_method=model_name,
                                                                       X_tabular=X_posttrauma,
                                                                       nominal_features=nominal_features,
                                                                       feature_extractor=feature_extractor_clinical,
                                                                       summaries=patient_summaries,
                                                                       y=y_posttrauma)

        """        
        # HGBC Concatenated (Tab. + Text Embeddings)
        (hgbc_conc_dataset, hgbc_conc_ml_method, hgbc_conc_emb_method,
         hgbc_conc_train_score, hgbc_conc_test_scores) = concat_lr_txt_emb(dataset_name=posttrauma_dataset,
                                                                           emb_method=model_name,
                                                                           X_tabular=X_posttrauma,
                                                                           nominal_features=nominal_features,
                                                                           feature_extractor=feature_extractor,
                                                                           summaries=patient_summaries,
                                                                           y=y_posttrauma)

        # Log. Reg. Concatenated (Tab. + RT Embeddings)
        (lr_conc_rte_dataset, lr_conc_rte_ml_method, lr_conc_rte_emb_method,
         lr_conc_rte_train_score, lr_conc_rte_test_scores) = concat_lr_txt_emb(dataset_name=posttrauma_dataset,
                                                                               emb_method=f"{model_name} + RTE",
                                                                               X_tabular=X_posttrauma,
                                                                               nominal_features=nominal_features,
                                                                               feature_extractor=feature_extractor,
                                                                               summaries=patient_summaries,
                                                                               y=y_posttrauma)

        # HGBC Concatenated (Tab. + RT Embeddings)
        (hgbc_conc_rte_dataset, hgbc_conc_rte_ml_method, hgbc_conc_rte_emb_method,
         hgbc_conc_rte_train_score, hgbc_conc_rte_test_scores) = concat_lr_txt_emb(dataset_name=posttrauma_dataset,
                                                                                   emb_method=f"{model_name} + RTE",
                                                                                   X_tabular=X_posttrauma,
                                                                                   nominal_features=nominal_features,
                                                                                   feature_extractor=feature_extractor,
                                                                                   summaries=patient_summaries,
                                                                                   y=y_posttrauma)"""

        # Todo: check
        save_results_to_csv(output_file=f"{model_name}_LR_conc_train.csv", dataset_name=lr_conc_dataset,
                            ml_method=lr_conc_ml_method, emb_method=lr_conc_emb_method, concatenation="no",
                            metrics=lr_conc_train_score, is_train=True)

        save_results_to_csv(output_file=f"{model_name}_LR_conc_test.csv", dataset_name=lr_conc_dataset,
                            ml_method=lr_conc_ml_method, emb_method=lr_conc_emb_method, concatenation="no",
                            metrics=lr_conc_test_scores, is_train=False)

        """ 
        save_results_to_csv(output_file=f"{model_name}_LR_rte_train.csv", dataset_name=lr_conc_rte_dataset,
                            ml_method=lr_conc_rte_ml_method, emb_method=lr_conc_rte_emb_method, concatenation="no",
                            metrics=lr_conc_rte_train_score, is_train=True)

        save_results_to_csv(output_file=f"{model_name}_LR_rte_test.csv", dataset_name=lr_conc_rte_dataset,
                            ml_method=lr_conc_rte_ml_method, emb_method=lr_conc_rte_emb_method, concatenation="no",
                            metrics=lr_conc_rte_test_scores, is_train=False)

        save_results_to_csv(output_file=f"{model_name}_HGBC_train.csv", dataset_name=hgbc_conc_dataset,
                            ml_method=hgbc_conc_ml_method, emb_method=hgbc_conc_emb_method, concatenation="no",
                            metrics=hgbc_conc_train_score, is_train=True)

        save_results_to_csv(output_file=f"{model_name}_HGBC_test.csv", dataset_name=hgbc_conc_dataset,
                            ml_method=hgbc_conc_ml_method, emb_method=hgbc_conc_emb_method, concatenation="no",
                            metrics=hgbc_conc_test_scores, is_train=False)

        save_results_to_csv(output_file=f"{model_name}_HGBC_rte_train.csv", dataset_name=hgbc_conc_rte_dataset,
                            ml_method=hgbc_conc_rte_ml_method, emb_method=hgbc_conc_rte_emb_method, concatenation="no",
                            metrics=hgbc_conc_rte_train_score, is_train=True)

        save_results_to_csv(output_file=f"{model_name}_HGBC_rte_test.csv", dataset_name=hgbc_conc_rte_dataset,
                            ml_method=hgbc_conc_rte_ml_method, emb_method=hgbc_conc_rte_emb_method, concatenation="no",
                            metrics=hgbc_conc_rte_test_scores, is_train=False)
        """
        labels_local = [
            f"{model_name} \n+ LogReg Table + Text Emb.",
            # f"{model_name} \n+ HGBC Table + Text Emb."
            # f"{model_name} \n+ LogReg Table + RT Emb",
            # f"{model_name} \n+ HGBC Table + RT Emb."
        ]
        train_scores_local = [
            lr_conc_train_score,
            # hgbc_conc_txt_train_score,
            # lr_conc_rte_train_score,
            # hgbc_conc_rte_train_score
        ]
        test_score_medians_local = [
            np.median(lr_conc_test_scores),
            # np.median(hgbc_conc_txt_test_scores),
            # np.median(lr_conc_rte_test_scores),
            # np.median(hgbc_conc_rte_test_scores)
        ]
        test_score_mins_local = [
            np.min(lr_conc_test_scores),
            # np.min(hgbc_conc_txt_test_scores),
            # np.min(lr_conc_rte_test_scores),
            # np.min(hgbc_conc_rte_test_scores)
        ]
        test_score_maxs_local = [
            np.max(lr_conc_test_scores),
            # np.max(hgbc_conc_txt_test_scores),
            # np.max(lr_conc_rte_test_scores),
            # np.max(hgbc_conc_rte_test_scores)
        ]

        # Convert to arrays
        train_scores_local = np.array(train_scores_local)
        test_score_medians_local = np.array(test_score_medians_local)
        test_score_mins_local = np.array(test_score_mins_local)
        test_score_maxs_local = np.array(test_score_maxs_local)

        plot_bar_chart(
            filename=f"{model_name}",
            labels=labels_local,
            train_scores=train_scores_local,
            test_score_medians=test_score_medians_local,
            test_score_mins=test_score_mins_local,
            test_score_maxs=test_score_maxs_local
        )
