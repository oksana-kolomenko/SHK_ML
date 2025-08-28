import numpy as np

from csv_saver import save_results_to_csv
from data_preps import load_features, load_labels, load_summaries
from helpers import concat_lr_rte, concat_hgbc_rte, concat_lr_txt_emb, concat_hgbc_txt_emb, hgbc_txt_emb, lr_txt_emb
from models import feature_extractor_e5_large_v2, feature_extractor_gist_embedding_v0, feature_extractor_gte_base, \
    feature_extractor_gte_base_en_v1_5, feature_extractor_gte_large, feature_extractor_bge_base_en_v1_5, \
    feature_extractor_bge_large_en_v1_5, feature_extractor_gist_large_embedding_v0, feature_extractor_sentence_t5_base, \
    feature_extractor_gtr_t5_base

#from helpers import concat_lr_rte, concat_hgbc_rte, lr_txt_emb, hgbc_txt_emb, concat_lr_txt_emb, concat_hgbc_txt_emb
from models import (feature_extractor_all_minilm_l6_v2, feature_extractor_bge_base_en_v1_5, feature_extractor_bge_large_en_v1_5,\
    feature_extractor_gist_large_embedding_v0, feature_extractor_bge_small_en_v1_5, \
    feature_extractor_gist_small_embedding_v0, feature_extractor_gte_small, feature_extractor_e5_small_v2, \
    feature_extractor_e5_base_v2, feature_extractor_stella_en_400M_v5,
                    feature_extractor_ember_v1)
from values import DatasetName

#from helpers_new import concat_hgbc_txt_emb
"""from models import feature_extractor_medembed_small_v0_1, feature_extractor_medembed_base_v0_1, \
    feature_extractor_gte_small, feature_extractor_gte_base, feature_extractor_gte_base_en_v1_5, \
    feature_extractor_gist_small_embedding_v0, feature_extractor_gist_embedding_v0, feature_extractor_electra_small, \
    feature_extractor_electra_base, feature_extractor_simsce_sup, feature_extractor_simsce_unsup, \
    feature_extractor_e5_small_v2, feature_extractor_e5_base_v2, feature_extractor_bge_small_en_v1_5, \
    feature_extractor_bge_base_en_v1_5, feature_extractor_clinical, feature_extractor_bert
"""
from models import (feature_extractor_gist_large_embedding_v0,
                    feature_extractor_bge_large_en_v1_5, feature_extractor_e5_large_v2)

from data_prep import mimic_nom_features
# from models import (feature_extractor_gte_large, feature_extractor_medembed_large_v0_1, feature_extractor_gte_large_en_v1_5)


def run_txt_emb():
    # === LUNGDISEASE ===
    """
    dataset = DatasetName.LUNG_DISEASE.value
    y = load_labels("y_lung_disease_data.csv")
    X = load_features("X_lung_disease_data.csv")
    X_metr = load_features("X_lungdisease_metrics.csv")
    all_summaries = load_summaries("_lung_disease_summaries.txt")
    nom_summaries = load_summaries("lungdisease_nom_summaries.txt")

    nominal_features = [
        'Gender',
        'Smoking Status',
        'Disease Type',
        'Treatment Type'
    ]"""

    # === CYBERSECURITY ===
    """dataset = DatasetName.CYBERSECURITY.value
    y = load_labels("y_cybersecurity_intrusion_data.csv")
    X = load_features("X_cybersecurity_intrusion_data.csv")
    X_metr = load_features("X_cybersecurity_metrics.csv")
    all_summaries = load_summaries("_cybersecurity_summaries.txt")
    nom_summaries = load_summaries("_cybersecurity_nom_summaries.txt")

    nominal_features = [
        'encryption_used',
        'browser_type',
        'protocol_type',
        'unusual_time_access'
    ]"""

    # === MIMIC ===
    task = "task_0"
    dataset = DatasetName.MIMIC_0.value

    X_train = load_features(f"mimic_data/X_train_{task}.csv")
    X_test = load_features(f"mimic_data/X_test_{task}.csv")

    X_train_metr = load_features(f"mimic_data/X_metr_train_{task}.csv")
    X_test_metr = load_features(f"mimic_data/X_metr_test_{task}.csv")

    train_summaries = load_summaries(f"mimic_data/mimic_{task}_train_summaries.txt")
    test_summaries = load_summaries(f"mimic_data/mimic_{task}_test_summaries.txt")

    train_nom_summaries = load_summaries(f"mimic_data/nominal_summaries/mimic_{task}_train_nom_summaries.txt")
    test_nom_summaries = load_summaries(f"mimic_data/nominal_summaries/mimic_{task}_test_nom_summaries.txt")

    y_train = load_labels(f"mimic_data/y_train_{task}.csv")
    y_test = load_labels(f"mimic_data/y_test_{task}.csv")

    nominal_features = mimic_nom_features(X=X_train)

    methods = {
        # all summaries, all features
        #"pca_conc1": {"X_train": X_train,
        #              "X_test": X_test,
        #              "train_summaries": train_summaries,
        #              "test_summaries": test_summaries,
        #              "conc": "conc1",
        #              "pca": True,
        #              "pca_str": "pca_"},

        # all summaries, metr features
        #"pca_conc2": {"X_train": X_train_metr,
        #              "X_test": X_test_metr,
        #              "train_summaries": train_summaries,
        #              "test_summaries": test_summaries,
        #              "conc": "conc2",
        #              "pca": True,
        #              "pca_str": "pca_"},

        # nom summaries, metr features
        "pca_conc3": {"X_train": X_train_metr,
                      "X_test": X_test_metr,
                      "train_summaries": train_nom_summaries,
                      "test_summaries": test_nom_summaries,
                      "conc": "conc3",
                      "pca": True,
                      "pca_str": "pca_"},

        # all summaries, all features
        #"conc1": {"X_train": X_train,
        #          "X_test": X_test,
        #          "train_summaries": train_summaries,
        #          "test_summaries": test_summaries,
        #          "conc": "conc1",
        #          "pca": False,
        #          "pca_str": ""},

        # all summaries, metr features
        #"conc2": {"X_train": X_train_metr,
        #          "X_test": X_test_metr,
        #          "train_summaries": train_summaries,
        #          "test_summaries": test_summaries,
        #          "conc": "conc2",
        #          "pca": False,
        #          "pca_str": ""},

        # nom summaries, metr features
        "conc3": {"X_train": X_train_metr,
                  "X_test": X_test_metr,
                  "train_summaries": train_nom_summaries,
                  "test_summaries": test_nom_summaries,
                  "conc": "conc3",
                  "pca": False,
                  "pca_str": ""}

    }
    text_feature = 'text'

    feature_extractors = {
        # All MiniLM L6 v2
        #"all_miniLM_L6_v2": feature_extractor_all_minilm_l6_v2,

        # Stella en 400m v5
        #"Stella-EN-400M-v5": feature_extractor_stella_en_400M_v5,

        # Ember v1
        #"ember_v1": feature_extractor_ember_v1,

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

        # GTE Models
        #"GTE-Small": feature_extractor_gte_small,
        #"GTE-Base": feature_extractor_gte_base,
        #"GTE-Base-EN-v1.5": feature_extractor_gte_base_en_v1_5,
        "GTE-Large": feature_extractor_gte_large,

        # GTR T5 Base
        "GTR_T5_Base": feature_extractor_gtr_t5_base,

        # Sentence T5 Base
        "sentence_t5_base": feature_extractor_sentence_t5_base,

        # Potion Models
        # "Potion-Base-2M": feature_extractor_potion_base_2M,
        # "Potion-Base-4M": feature_extractor_potion_base_4M,
        # "Potion-Base-8M": feature_extractor_potion_base_8M,

        ####### jetzt nicht ################
        # Clinical Longformer (done)
        # "Clinical-Longformer": feature_extractor_clinical,

        # BERT (half done)
        # "BERT-Base": feature_extractor_bert,

        # ELECTRA (half done)
        # "ELECTRA-Small": feature_extractor_electra_small,
        # "ELECTRA-Base": feature_extractor_electra_base,
        # "ELECTRA-Large": feature_extractor_electra_large,

        # SimSCE (done)
        # "SimSCE-Sup": feature_extractor_simsce_sup,
        # "SimSCE-Unsup": feature_extractor_simsce_unsup,

        # MedEmbed Models (problem)
        # "MedEmbed-Small-v0.1": feature_extractor_medembed_small_v0_1,
        # "MedEmbed-Base-v0.1": feature_extractor_medembed_base_v0_1,
        # "MedEmbed-Large-v0.1": feature_extractor_medembed_large_v0_1,

        # "GTE-Large-EN-v1.5": feature_extractor_gte_large_en_v1_5,

        # modernbert-embed-base
        # "modernbert_embed_base": feature_extractor_mbert_embed_base,

        # GTE modernbert base
        # "gte_modernbert_base": feature_extractor_gte_mbert_base,
    }

    for model_name, feature_extractor in feature_extractors.items():
        #######################
        ### no PCA, no CONC ###
        #######################

        # Logistic Regression
        """
        (lr_txt_dataset, lr_txt_ml_method, lr_txt_emb_method, lr_txt_concatenation, lr_txt_best_params,
         lr_txt_pca_components, lr_txt_train_score, lr_txt_test_scores) = lr_txt_emb(
            dataset_name=dataset, emb_method=model_name,
            feature_extractor=feature_extractor, max_iter=10000,
            train_summaries=train_summaries, test_summaries=test_summaries, y_train=y_train,
            y_test=y_test, pca=False)

        save_results_to_csv(output_file=f"{dataset}_{model_name}_LR_train.csv", dataset_name=lr_txt_dataset,
                            ml_method=lr_txt_ml_method, emb_method=lr_txt_emb_method, concatenation=lr_txt_concatenation,
                            best_params=lr_txt_best_params, pca_n_comp=lr_txt_pca_components,
                            metrics=lr_txt_train_score, is_train=True)

        save_results_to_csv(output_file=f"{dataset}_{model_name}_LR_test.csv", dataset_name=lr_txt_dataset,
                            ml_method=lr_txt_ml_method, emb_method=lr_txt_emb_method, concatenation=lr_txt_concatenation,
                            best_params=lr_txt_best_params, pca_n_comp=lr_txt_pca_components,
                            metrics=lr_txt_test_scores, is_train=False)

        # HGBC
        
        (hgbc_txt_dataset, hgbc_txt_ml_method, hgbc_txt_emb_method, hgbc_txt_conc, hgbc_best_params, hgbc_pca_comp,
         hgbc_txt_train_score, hgbc_txt_test_scores) \
            = hgbc_txt_emb(dataset_name=dataset,
                           emb_method=model_name,
                           feature_extractor=feature_extractor,
                           train_summaries=train_summaries,
                           test_summaries=test_summaries,
                           y_train=y_train, y_test=y_test,
                           pca=False)

        save_results_to_csv(output_file=f"{dataset}_{model_name}_HGBC_train.csv",
                            dataset_name=hgbc_txt_dataset,
                            ml_method=hgbc_txt_ml_method,
                            emb_method=hgbc_txt_emb_method,
                            concatenation=hgbc_txt_conc,
                            best_params=hgbc_best_params,
                            pca_n_comp=hgbc_pca_comp,
                            metrics=hgbc_txt_train_score,
                            is_train=True)

        save_results_to_csv(output_file=f"{dataset}_{model_name}_HGBC_test.csv",
                            dataset_name=hgbc_txt_dataset,
                            ml_method=hgbc_txt_ml_method,
                            emb_method=hgbc_txt_emb_method,
                            concatenation=hgbc_txt_conc,
                            best_params=hgbc_best_params,
                            pca_n_comp=hgbc_pca_comp,
                            metrics=hgbc_txt_test_scores,
                            is_train=False)


        ####################
        ### PCA, no CONC ###
        ####################

        # Logistic Regression

        (lr_txt_dataset, lr_txt_ml_method, lr_txt_emb_method, lr_txt_concatenation, lr_txt_best_params,
         lr_txt_pca_components, lr_txt_train_score, lr_txt_test_scores) = lr_txt_emb(
            dataset_name=dataset, emb_method=model_name,
            feature_extractor=feature_extractor, max_iter=10000,
            train_summaries=train_summaries, test_summaries=test_summaries, y_train=y_train,
            y_test=y_test, pca=True)

        save_results_to_csv(output_file=f"{dataset}_{model_name}_LR_pca_train.csv", dataset_name=lr_txt_dataset,
                            ml_method=lr_txt_ml_method, emb_method=lr_txt_emb_method, concatenation=lr_txt_concatenation,
                            best_params=lr_txt_best_params, pca_n_comp=lr_txt_pca_components,
                            metrics=lr_txt_train_score, is_train=True)

        save_results_to_csv(output_file=f"{dataset}_{model_name}_LR_pca_test.csv", dataset_name=lr_txt_dataset,
                            ml_method=lr_txt_ml_method, emb_method=lr_txt_emb_method, concatenation=lr_txt_concatenation,
                            best_params=lr_txt_best_params, pca_n_comp=lr_txt_pca_components,
                            metrics=lr_txt_test_scores, is_train=False)
        

        # HGBC
        (hgbc_txt_dataset, hgbc_txt_ml_method, hgbc_txt_emb_method, hgbc_txt_conc, hgbc_best_params, hgbc_pca_comp,
         hgbc_txt_train_score, hgbc_txt_test_scores) \
            = hgbc_txt_emb(dataset_name=dataset,
                           emb_method=model_name,
                           feature_extractor=feature_extractor,
                           train_summaries=train_summaries,
                           test_summaries=test_summaries,
                           y_train=y_train, y_test=y_test,
                           pca=True)

        save_results_to_csv(output_file=f"{dataset}_{model_name}_HGBC_pca_train.csv",
                            dataset_name=hgbc_txt_dataset,
                            ml_method=hgbc_txt_ml_method,
                            emb_method=hgbc_txt_emb_method,
                            concatenation=hgbc_txt_conc,
                            best_params=hgbc_best_params,
                            pca_n_comp=hgbc_pca_comp,
                            metrics=hgbc_txt_train_score,
                            is_train=True)

        save_results_to_csv(output_file=f"{dataset}_{model_name}_HGBC_pca_test.csv",
                            dataset_name=hgbc_txt_dataset,
                            ml_method=hgbc_txt_ml_method,
                            emb_method=hgbc_txt_emb_method,
                            concatenation=hgbc_txt_conc,
                            best_params=hgbc_best_params,
                            pca_n_comp=hgbc_pca_comp,
                            metrics=hgbc_txt_test_scores,
                            is_train=False)
                            """

        for method_name, attributes in methods.items():
            #################
            ### PCA, CONC ###
            #################

            conc_art = attributes.get("conc")
            X_train = attributes.get("X_train")
            X_test = attributes.get("X_test")
            train_summaries = attributes.get("train_summaries")
            test_summaries = attributes.get("test_summaries")
            pca = attributes.get("pca")
            pca_str = attributes.get("pca_str")

            # Logistic Regression conc (pca)
            (lr_conc_dataset, lr_conc_ml_method, lr_conc_emb_method,
             lr_conc_yesno, lr_best_params, lr_pca_components, lr_conc_train_score,
             lr_conc_test_scores) = concat_lr_txt_emb(
                dataset_name=dataset,
                emb_method=model_name,
                feature_extractor=feature_extractor,

                train_summaries=train_summaries,
                test_summaries=test_summaries,

                X_tab_train=X_train,
                X_tab_test=X_test,

                y_train=y_train,
                y_test=y_test,

                nominal_features=nominal_features,

                text_feature_column_name=text_feature,
                concatenation=conc_art,
                imp_max_iter=30, class_max_iter=10000, pca=pca)

            save_results_to_csv(output_file=f"{dataset}_{model_name}_LR_{conc_art}_{pca_str}train.csv",
                                dataset_name=lr_conc_dataset,
                                ml_method=lr_conc_ml_method,
                                emb_method=lr_conc_emb_method,
                                concatenation=lr_conc_yesno,
                                best_params=lr_best_params,
                                pca_n_comp=lr_pca_components,
                                metrics=lr_conc_train_score,
                                is_train=True)

            save_results_to_csv(output_file=f"{dataset}_{model_name}_LR_{conc_art}_{pca_str}test.csv",
                                dataset_name=lr_conc_dataset,
                                ml_method=lr_conc_ml_method,
                                emb_method=lr_conc_emb_method,
                                concatenation=lr_conc_yesno,
                                best_params=lr_best_params,
                                pca_n_comp=lr_pca_components,
                                metrics=lr_conc_test_scores,
                                is_train=False)

            # HGBC conc (pca)
            """
            (concat_hgbc_dataset, concat_hgbc_ml_method, concat_hgbc_emb_method,
             hgbc_conc_yesno, hgbc_best_params, hgbc_pca_components, hgbc_conc_train_score,
             hgbc_conc_test_scores) = concat_hgbc_txt_emb(
                dataset_name=dataset,
                emb_method=model_name,
                feature_extractor=feature_extractor,
                raw_text_summaries=summaries,
                X_tabular=X, y=y,
                text_feature_column_name=text_feature,
                concatenation=conc_art, pca=pca, nominal_features=nominal_features)

            save_results_to_csv(output_file=f"{dataset}_{model_name}_HGBC_{conc_art}_{pca_str}train.csv",
                                dataset_name=concat_hgbc_dataset,
                                ml_method=concat_hgbc_ml_method,
                                emb_method=concat_hgbc_emb_method,
                                concatenation=hgbc_conc_yesno,
                                best_params=hgbc_best_params,
                                pca_n_comp=hgbc_pca_components,
                                metrics=hgbc_conc_train_score,
                                is_train=True)

            save_results_to_csv(output_file=f"{dataset}_{model_name}_HGBC_{conc_art}_{pca_str}test.csv",
                                dataset_name=concat_hgbc_dataset,
                                ml_method=concat_hgbc_ml_method,
                                emb_method=concat_hgbc_emb_method,
                                concatenation=hgbc_conc_yesno,
                                best_params=hgbc_best_params,
                                pca_n_comp=hgbc_pca_components,
                                metrics=hgbc_conc_test_scores,
                                is_train=False)
            """


def run_pca_rte():
    posttrauma_dataset = DatasetName.POSTTRAUMA.value
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
    (lr_rt_conc_dataset, lr_rt_conc_ml_method, lr_rt_conc_emb_method,
     lr_rt_conc_yesno, lr_rt_best_params, lr_rt_pca_components,
     lr_rt_conc_train_score, lr_rt_conc_test_scores) = concat_lr_rte(
        dataset_name=posttrauma_dataset,
        X_tabular=X_posttrauma, y=y_posttrauma,
        nominal_features=nominal_features, pca_n_comp=0.95,
        n_repeats=10, class_max_iter=10000, imp_max_iter=50)

    # todo:save train&test results as list and iterate
    save_results_to_csv(output_file=f"LR_rte_conc__pca_095_train.csv",
                        dataset_name=lr_rt_conc_dataset,
                        ml_method=lr_rt_conc_ml_method,
                        emb_method=lr_rt_conc_emb_method,
                        concatenation=lr_rt_conc_yesno,
                        best_params=lr_rt_best_params,
                        pca_n_comp=lr_rt_pca_components,
                        metrics=lr_rt_conc_train_score,
                        is_train=True)

    save_results_to_csv(output_file=f"LR_rte_conc_pca_095_test.csv",
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
                                                                             pca_n_comp=0.95)

    save_results_to_csv(output_file=f"HGBC_rte_conc_pca_095_train.csv", dataset_name=hgbc_conc_rte_dataset,
                        ml_method=hgbc_conc_rte_ml_method, emb_method=hgbc_conc_rte_emb_method,
                        concatenation=hgbc_rte_conc, metrics=hgbc_conc_rte_train_score,
                        best_params=hgbc_rte_best_params, pca_n_comp=hgbc_rte_pca_n_comp, is_train=True)

    save_results_to_csv(output_file=f"HGBC_rte_conc_pca_095_test.csv", dataset_name=hgbc_conc_rte_dataset,
                        ml_method=hgbc_conc_rte_ml_method, emb_method=hgbc_conc_rte_emb_method,
                        concatenation=hgbc_rte_conc, metrics=hgbc_conc_rte_test_scores,
                        best_params=hgbc_rte_best_params, pca_n_comp=hgbc_rte_pca_n_comp, is_train=False)