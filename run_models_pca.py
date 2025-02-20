import numpy as np

from csv_saver import save_results_to_csv
from helpers import load_labels, load_summaries, concat_lr_txt_emb, load_features, hgbc_txt_emb, lr_txt_emb
from bar_plotting import plot_bar_chart
from helpers_new import concat_hgbc_txt_emb
from models import feature_extractor_gte_large_en_v1_5, feature_extractor_gte_large
#from models import


from values import Dataset


def run_pca_txt_emb():
    posttrauma_dataset = Dataset.POSTTRAUMA.value

    # load features and labels
    patient_summaries = load_summaries()
    y_posttrauma = load_labels()
    X_posttrauma = load_features()

    nominal_features = [
        'gender_birth',
        'ethnic_group',
        'education_age',
        'working_at_baseline',
        'penetrating_injury'
    ]

    text_feature = 'text'

    feature_extractors = {
        # Clinical Longformer (done)
        #"Clinical-Longformer": feature_extractor_clinical,

        # BERT (half done)
        #"BERT": feature_extractor_bert,

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
        #"MedEmbed-Small-v0.1": feature_extractor_medembed_small_v0_1, # (problem)
        #"MedEmbed-Base-v0.1": feature_extractor_medembed_base_v0_1, # (problem)
        #"MedEmbed-Large-v0.1": feature_extractor_medembed_large_v0_1, # (problem)

        # Potion Models
        #"Potion-Base-2M": feature_extractor_potion_base_2M,
        #"Potion-Base-4M": feature_extractor_potion_base_4M,
        # "Potion-Base-8M": feature_extractor_potion_base_8M,

        # GTE Models
        #"GTE-Small": feature_extractor_gte_small,  # (done)
        #"GTE-Base": feature_extractor_gte_base,  # (done)
        #"GTE-Base-EN-v1.5": feature_extractor_gte_base_en_v1_5, #(ready)
        "GTE-Large": feature_extractor_gte_large,  # (done)
        "GTE-Large-EN-v1.5": feature_extractor_gte_large_en_v1_5, # (ready)

        # Stella Model
        # "Stella-EN-400M-v5": feature_extractor_stella_en_400M_v5 # (not ready)
    }

    for model_name, feature_extractor in feature_extractors.items():

        # Die Methoden müssen möglicherweise noch an pca angepasst werden
        # Logistic Regression
        # no concatenation

        (lr_txt_dataset, lr_txt_ml_method, lr_txt_emb_method, lr_txt_concatenation, lr_txt_best_params,
         lr_txt_pca_components, lr_txt_train_score, lr_txt_test_scores) = lr_txt_emb(
            dataset_name=posttrauma_dataset, n_components=35, emb_method=model_name,
            feature_extractor=feature_extractor, max_iter=10000, n_repeats=10,
            raw_text_summaries=patient_summaries, y=y_posttrauma)

        save_results_to_csv(output_file=f"{model_name}_LR_pca_35_train.csv", dataset_name=lr_txt_dataset,
                            ml_method=lr_txt_ml_method, emb_method=lr_txt_emb_method, concatenation=lr_txt_concatenation,
                            best_params=lr_txt_best_params, pca_n_comp=lr_txt_pca_components,
                            metrics=lr_txt_train_score, is_train=True)

        save_results_to_csv(output_file=f"{model_name}_LR_pca_35_test.csv", dataset_name=lr_txt_dataset,
                            ml_method=lr_txt_ml_method, emb_method=lr_txt_emb_method, concatenation=lr_txt_concatenation,
                            best_params=lr_txt_best_params, pca_n_comp=lr_txt_pca_components,
                            metrics=lr_txt_test_scores, is_train=False)

        # Logistic Regression
        # concatenation

        """(lr_conc_dataset, lr_conc_ml_method, lr_conc_emb_method,
         lr_conc_yesno, lr_best_params, lr_pca_components, lr_conc_train_score,
         lr_conc_test_scores) = concat_lr_txt_emb(
            dataset_name=posttrauma_dataset,
            emb_method=model_name,
            feature_extractor=feature_extractor,
            raw_text_summaries=patient_summaries,
            X_tabular=X_posttrauma, y=y_posttrauma,
            nominal_features=nominal_features,
            text_feature_column_name=text_feature,
            imp_max_iter=30, class_max_iter=10000,
            n_components=35, n_repeats=10)

        # todo:save train&test results as list and iterate
        save_results_to_csv(output_file=f"{model_name}_LR_conc_pca_35_train.csv",
                            dataset_name=lr_conc_dataset,
                            ml_method=lr_conc_ml_method,
                            emb_method=lr_conc_emb_method,
                            concatenation=lr_conc_yesno,
                            best_params=lr_best_params,
                            pca_n_comp=lr_pca_components,
                            metrics=lr_conc_train_score,
                            is_train=True)

        save_results_to_csv(output_file=f"{model_name}_LR_conc_pca_35_test.csv",
                            dataset_name=lr_conc_dataset,
                            ml_method=lr_conc_ml_method,
                            emb_method=lr_conc_emb_method,
                            concatenation=lr_conc_yesno,
                            best_params=lr_best_params,
                            pca_n_comp=lr_pca_components,
                            metrics=lr_conc_test_scores,
                            is_train=False)"""

        """(concat_hgbcdataset, concat_hgbc_ml_method, concat_hgbc_emb_method,
         lr_conc_yesno, lr_best_params, lr_pca_components, lr_conc_train_score,
         lr_conc_test_scores) = concat_hgbc_txt_emb(
            dataset_name=posttrauma_dataset,
            emb_method=model_name,
            feature_extractor=feature_extractor,
            raw_text_summaries=patient_summaries,
            X_tabular=X_posttrauma, y=y_posttrauma,
            nominal_features=nominal_features,
            text_feature_column_name=text_feature,
            imp_max_iter=30, class_max_iter=10000,
            n_components=35, n_repeats=10)

        # todo:save train&test results as list and iterate
        save_results_to_csv(output_file=f"{model_name}_LR_conc_pca_35_train.csv",
                            dataset_name=lr_conc_dataset,
                            ml_method=lr_conc_ml_method,
                            emb_method=lr_conc_emb_method,
                            concatenation=lr_conc_yesno,
                            best_params=lr_best_params,
                            pca_n_comp=lr_pca_components,
                            metrics=lr_conc_train_score,
                            is_train=True)

        save_results_to_csv(output_file=f"{model_name}_LR_conc_pca_35_test.csv",
                            dataset_name=lr_conc_dataset,
                            ml_method=lr_conc_ml_method,
                            emb_method=lr_conc_emb_method,
                            concatenation=lr_conc_yesno,
                            best_params=lr_best_params,
                            pca_n_comp=lr_pca_components,
                            metrics=lr_conc_test_scores,
                            is_train=False)"""

        """# HGBC
        (hgbc_txt_dataset, hgbc_txt_ml_method, hgbc_txt_emb_method, hgbc_txt_conc, hgbc_best_params, hgbc_pca_comp,
         hgbc_txt_train_score, hgbc_txt_test_scores) \
            = hgbc_txt_emb(dataset_name=posttrauma_dataset, emb_method=model_name, n_components=35,
                           n_repeats=10, feature_extractor=model_name, summaries=patient_summaries, y=y_posttrauma)

        save_results_to_csv(output_file=f"{model_name}_HGBC_pca_train.csv",
                            dataset_name=hgbc_txt_dataset,
                            ml_method=hgbc_txt_ml_method,
                            emb_method=hgbc_txt_emb_method,
                            concatenation=hgbc_txt_conc,
                            best_params=hgbc_best_params,
                            pca_n_comp=hgbc_pca_comp,
                            metrics=hgbc_txt_train_score,
                            is_train=True)

        save_results_to_csv(output_file=f"{model_name}_HGBC_pca_test.csv",
                            dataset_name=hgbc_txt_dataset,
                            ml_method=hgbc_txt_ml_method,
                            emb_method=hgbc_txt_emb_method,
                            concatenation=hgbc_txt_conc,
                            best_params=hgbc_best_params,
                            pca_n_comp=hgbc_pca_comp,
                            metrics=hgbc_txt_test_scores,
                            is_train=False)"""

        """
        # Geht gerade nicht, da scores enthalten mehrere Metrics        
        labels_local = [
            f"{model_name} \n+ Log Reg + PCA",
            f"{model_name} \n+ HGBC + PCA"
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
            filename=f"{model_name}_pca",
            labels=labels_local,
            train_scores=train_scores_local,
            test_score_medians=test_score_medians_local,
            test_score_mins=test_score_mins_local,
            test_score_maxs=test_score_maxs_local
        )"""
