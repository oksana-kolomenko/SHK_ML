import numpy as np

from csv_saver import save_results_to_csv
from helpers import (load_labels, load_features, load_summaries, logistic_regression,
                     lr_ran_tree_emb, lr_txt_emb, hgbc, hgbc_txt_emb, concat_lr_tab_txtemb)
from bar_plotting import plot_bar_chart
from models import feature_extractor_clinical, feature_extractor_electra_small, feature_extractor_electra_large, feature_extractor_electra_base, \
    feature_extractor_bert
from values import Dataset


# from models import feature_extractor_clinical


def run_all_models():
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

    feature_extractors = {
        # Clinical Longformer (done)
        "Clinical-Longformer": feature_extractor_clinical,

        # BERT (half done)
        #"BERT": feature_extractor_bert,

        # ELECTRA (half done)
        # "ELECTRA-Small": feature_extractor_electra_small,
        #"ELECTRA-Base": feature_extractor_electra_base,
        #"ELECTRA-Large": feature_extractor_electra_large,

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

    ########################
    ###### TABLE DATA ######
    ########################

    # 1. logistic regression (no embedding), dataset: posttrauma
    log_reg_dataset, log_reg_ml_method, log_reg_emb_method, log_reg_train_score, log_reg_test_scores = \
        logistic_regression(dataset=posttrauma_dataset, X=X_posttrauma, y=y_posttrauma,  # test
                            nominal_features=nominal_features)

    save_results_to_csv( # anpassen
    dataset_name=log_reg_dataset,
    ml_method=log_reg_ml_method,
    output_file="no_emb_log_reg_train.csv")

    # test ...

    # 2. random trees embedding
    (lr_rt_dataset, lr_rt_ml_method, lr_rt_emb_method, lr_rt_concatenation, log_reg_rt_emb_train_score,
     log_reg_rt_emb_test_scores) = lr_ran_tree_emb(dataset = posttrauma_dataset,X=X_posttrauma, y=y_posttrauma,
                                                   nominal_features=nominal_features)
    save_results_to_csv( # anpassen
    dataset_name=lr_rt_dataset,
    ml_method=lr_rt_ml_method,
    output_file="no_emb_log_reg_train.csv")

    # test ...

    
    # 3. hgbc (no embedding)
    hgbc_train_score, hgbc_test_scores = \
        hgbc(X=X_posttrauma, y=y_posttrauma, nominal_features=nominal_features)

    # 4. random trees embedding + hgbc
    hgbc_rt_emb_train_score, hgbc_rt_emb_test_scores = \
        lr_ran_tree_emb(X=X_posttrauma, y=y_posttrauma, nominal_features=nominal_features)

    labels_local = [
            f"Logistic Regression",
            f"LogReg + RTE"
            f"HGBC",
            f"HGBC + RTE"
        ]
    train_scores_local = [
            log_reg_train_score,
            log_reg_rt_emb_train_score,
            hgbc_train_score,
            hgbc_rt_emb_train_score
        ]
    test_score_medians_local = [
            np.median(log_reg_test_scores),
            np.median(log_reg_rt_emb_test_scores),
            np.median(hgbc_test_scores),
            np.median(hgbc_rt_emb_test_scores)
        ]
    test_score_mins_local = [
            np.min(log_reg_test_scores),
            np.min(log_reg_rt_emb_test_scores),
            np.min(hgbc_test_scores),
            np.min(hgbc_rt_emb_test_scores)
        ]
    test_score_maxs_local = [
            np.max(log_reg_test_scores),
            np.max(log_reg_rt_emb_test_scores),
            np.max(hgbc_test_scores),
            np.max(hgbc_rt_emb_test_scores)
        ]

    # Convert to arrays
    train_scores_local = np.array(train_scores_local)
    test_score_medians_local = np.array(test_score_medians_local)
    test_score_mins_local = np.array(test_score_mins_local)
    test_score_maxs_local = np.array(test_score_maxs_local)

    plot_bar_chart(
            filename=f"no_and_rte_embeddings",
            labels=labels_local,
            train_scores=train_scores_local,
            test_score_medians=test_score_medians_local,
            test_score_mins=test_score_mins_local,
            test_score_maxs=test_score_maxs_local
    )


    ########################
    ###### EMBEDDINGS ######
    ########################

    patient_summaries = load_summaries()

    # Calculate results for each model
    for model_name, feature_extractor in feature_extractors.items():
        # Logistic Regression
        lr_txt_train_score, lr_txt_test_scores = lr_txt_emb(
            feature_extractor=feature_extractor, summaries=patient_summaries, y=y_posttrauma)

        # HGBC
        hgbc_txt_train_score, hgbc_txt_test_scores = hgbc_txt_emb(
            feature_extractor=feature_extractor, summaries=patient_summaries, y=y_posttrauma)

        # Log. Reg. Concatenated (Tab. + Text Embeddings)
        lr_conc_txt_train_score, lr_conc_txt_test_scores = concat_lr_tab_txtemb(X_tabular=X_posttrauma,
            nominal_features=nominal_features, feature_extractor=feature_extractor, summaries=patient_summaries, y=y_posttrauma)

        # HGBC Concatenated (Tab. + Text Embeddings)
        hgbc_conc_txt_train_score, hgbc_conc_txt_test_scores = concat_lr_tab_txtemb(X_tabular=X_posttrauma,
            nominal_features=nominal_features, feature_extractor=feature_extractor, summaries=patient_summaries, y=y_posttrauma)

        # Log. Reg. Concatenated (Tab. + RT Embeddings)
        lr_conc_rte_train_score, lr_conc_rte_test_scores = concat_lr_tab_txtemb(X_tabular=X_posttrauma,
            nominal_features=nominal_features, feature_extractor=feature_extractor, summaries=patient_summaries, y=y_posttrauma)

        # HGBC Concatenated (Tab. + RT Embeddings)
        hgbc_conc_rte_train_score, hgbc_conc_rte_test_scores = concat_lr_tab_txtemb(X_tabular=X_posttrauma,
            nominal_features=nominal_features, feature_extractor=feature_extractor, summaries=patient_summaries, y=y_posttrauma)

        labels_local = [
            f"{model_name} \n+ Logistic Regression",
            f"{model_name} \n+ HGBC"
            f"{model_name} \n+ LogReg Table + Text Emb.",
            f"{model_name} \n+ HGBC Table + Text Emb."
            f"{model_name} \n+ LogReg Table + RT Emb",
            f"{model_name} \n+ HGBC Table + RT Emb."
        ]
        train_scores_local = [
            lr_txt_train_score,
            hgbc_txt_train_score,
            lr_conc_txt_train_score,
            hgbc_conc_txt_train_score,
            lr_conc_rte_train_score,
            hgbc_conc_rte_train_score
        ]
        test_score_medians_local = [
            np.median(lr_txt_test_scores),
            np.median(hgbc_txt_test_scores),
            np.median(lr_conc_txt_test_scores),
            np.median(hgbc_conc_txt_test_scores),
            np.median(lr_conc_rte_test_scores),
            np.median(hgbc_conc_rte_test_scores)
        ]
        test_score_mins_local = [
            np.min(lr_txt_test_scores),
            np.min(hgbc_txt_test_scores),
            np.min(lr_conc_txt_test_scores),
            np.min(hgbc_conc_txt_test_scores),
            np.min(lr_conc_rte_test_scores),
            np.min(hgbc_conc_rte_test_scores)
        ]
        test_score_maxs_local = [
            np.max(lr_txt_test_scores),
            np.max(hgbc_txt_test_scores),
            np.max(lr_conc_txt_test_scores),
            np.max(hgbc_conc_txt_test_scores),
            np.max(lr_conc_rte_test_scores),
            np.max(hgbc_conc_rte_test_scores)
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
