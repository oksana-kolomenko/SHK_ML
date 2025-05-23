import numpy as np

from csv_saver import save_results_to_csv
from data_preps import load_summaries, load_labels
from helpers import lr_txt_emb, hgbc_txt_emb
from bar_plotting import plot_bar_chart
from models import feature_extractor_ember_v1, feature_extractor_stella_en_400M_v5, feature_extractor_all_minilm_l6_v2, \
    feature_extractor_gtr_t5_base, feature_extractor_sentence_t5_base
from values import DatasetName


def run_models_on_txt_emb():
    # posttrauma_dataset = Dataset.POSTTRAUMA.value
    # posttrauma_summaries = load_summaries("Summaries.txt")

    cybersecurity_dataset = DatasetName.CYBERSECURITY.value
    cybersecurity_summaries = load_summaries("cybersecurity_summaries.txt")
    y_cybersecurity = load_labels("y_cybersecurity_intrusion_data.csv")

    print('Starting to create FE')
    feature_extractors = {
        # Clinical Longformer (done)
        # "Clinical-Longformer": feature_extractor_clinical,

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
        # "MedEmbed-Small-v0.1": feature_extractor_medembed_small_v0_1,
        # "MedEmbed-Base-v0.1": feature_extractor_medembed_base_v0_1,
        # "MedEmbed-Large-v0.1": feature_extractor_medembed_large_v0_1,

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
        #"Stella-EN-400M-v5": feature_extractor_stella_en_400M_v5, # only GPU

        # All MiniLM L6 v2
        "all_miniLM_L6_v2": feature_extractor_all_minilm_l6_v2, # runs

        # GTR T5 Base
        "gtr-t5-base": feature_extractor_gtr_t5_base, #

        # Sentence T5 Base
        "sentence_t5_base": feature_extractor_sentence_t5_base,

        # modernbert-embed-base
        #"modernbert_embed_base": feature_extractor_mbert_embed_base,

        # GTE modernbert base
        #"gte_modernbert_base": feature_extractor_gte_mbert_base,

        # Ember v1
        "ember_v1": feature_extractor_ember_v1
    }

    ###### TEXT EMBEDDINGS ######
    # print(f"len summaries (run mod): {all_summaries}")
    # Calculate results for each model
    for model_name, feature_extractor in feature_extractors.items():
        print("Started For-Loop.")
        # HGBC
        (hgbc_txt_dataset, hgbc_txt_ml_method, hgbc_txt_emb_method, hgbc_txt_conc, hgbc_txt_best_params,
         hgbc_txt_pca_components, hgbc_txt_train_score, hgbc_txt_test_scores) = hgbc_txt_emb(
            dataset_name=cybersecurity_dataset,
            emb_method=model_name,
            n_components=None,
            n_repeats=10,
            feature_extractor=feature_extractor,
            summaries=cybersecurity_summaries,
            y=y_cybersecurity)

        save_results_to_csv(output_file=f"{model_name}_HGBC_train_{cybersecurity_dataset}.csv", dataset_name=hgbc_txt_dataset,
                            ml_method=hgbc_txt_ml_method, emb_method=hgbc_txt_emb_method, concatenation="no",
                            best_params=hgbc_txt_best_params, pca_n_comp=hgbc_txt_pca_components,
                            metrics=hgbc_txt_train_score, is_train=True)

        save_results_to_csv(output_file=f"{model_name}_HGBC_test{cybersecurity_dataset}.csv", dataset_name=hgbc_txt_dataset,
                            ml_method=hgbc_txt_ml_method, emb_method=hgbc_txt_emb_method, concatenation="no",
                            best_params=hgbc_txt_best_params, pca_n_comp=hgbc_txt_pca_components,
                            metrics=hgbc_txt_test_scores, is_train=False)
        # Logistic Regression
        (lr_txt_dataset, lr_txt_ml_method, lr_txt_emb_method, lr_txt_conc, lr_txt_best_params,
         lr_txt_pca_components, lr_txt_train_score, lr_txt_test_scores) = lr_txt_emb(
            dataset_name=cybersecurity_dataset,
            n_components=None,
            emb_method=model_name,
            #feature_extractor=feature_extractor, max_iter=100, n_repeats=1,
            feature_extractor=feature_extractor,
            max_iter=10000,
            n_repeats=10,
            raw_text_summaries=cybersecurity_summaries,
            y=y_cybersecurity)

        save_results_to_csv(output_file=f"{model_name}_LR_train_{cybersecurity_dataset}.csv", dataset_name=lr_txt_dataset,
                            ml_method=lr_txt_ml_method, emb_method=lr_txt_emb_method, concatenation=lr_txt_conc,
                            best_params=lr_txt_best_params, pca_n_comp=lr_txt_pca_components,
                            metrics=lr_txt_train_score, is_train=True)

        save_results_to_csv(output_file=f"{model_name}_LR_test_{cybersecurity_dataset}.csv", dataset_name=lr_txt_dataset,
                            ml_method=lr_txt_ml_method, emb_method=lr_txt_emb_method, concatenation=lr_txt_conc,
                            best_params=lr_txt_best_params, pca_n_comp=lr_txt_pca_components,
                            metrics=lr_txt_test_scores, is_train=False)

        """labels_local = [
            f"{model_name} \n+ Logistic Regression",
            f"{model_name} \n+ HGBC"
        ]
        train_scores_local = [
            lr_txt_train_score,
            hgbc_txt_train_score
        ]
        test_score_medians_local = [
            np.median(lr_txt_test_scores),
            np.median(hgbc_txt_test_scores)
        ]
        test_score_mins_local = [
            np.min(lr_txt_test_scores),
            np.min(hgbc_txt_test_scores)
        ]
        test_score_maxs_local = [
            np.max(lr_txt_test_scores),
            np.max(hgbc_txt_test_scores)
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
        )"""
