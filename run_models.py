import numpy as np
from helpers import (load_labels, load_features, load_summaries, logistic_regression,
                     lr_ran_tree_emb, lr_txt_emb, hgbc, hgbc_txt_emb, combine_lr_tab_emb)
from bar_plotting import plot_bar_chart
from models import feature_extractor_clinical, feature_extractor_electra_small, feature_extractor_electra_large, feature_extractor_electra_base, \
    feature_extractor_bert


# from models import feature_extractor_clinical


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
    """
    ########################
    ###### TABLE DATA ######
    ########################

    # 1. logistic regression (no embedding)
    lg_reg_train_score, lg_reg_test_scores = \
        logistic_regression(X=X, y=y, nominal_features=nominal_features)

    # 2. random trees embedding
    lg_reg_rt_emb_train_score, lg_reg_rt_emb_test_scores = \
        lr_ran_tree_emb(X=X, y=y, nominal_features=nominal_features)
    
    # 3. hgbc (no embedding)
    hgbc_train_score, hgbc_test_scores = \
        hgbc(X=X, y=y, nominal_features=nominal_features)

    # 4. random trees embedding + hgbc
    hgbc_rt_emb_train_score, hgbc_rt_emb_test_scores = \
        lr_ran_tree_emb(X=X, y=y, nominal_features=nominal_features)"""

    #######################
    ###### TEXT DATA ######
    #######################

    patient_summaries = load_summaries()

    labels = []
    train_scores = []
    test_score_medians = []
    test_score_mins = []
    test_score_maxs = []

    # Calculate results for each model
    for model_name, feature_extractor in feature_extractors.items():
        # Logistic Regression
        #lr_train_score, lr_test_scores = lr_txt_emb(
        #    feature_extractor=feature_extractor, summaries=patient_summaries, y=y
        #)

        # Log. Reg. Concatenated
        lr_train_score, lr_test_scores = combine_lr_tab_emb(X_tabular=X, nominal_features=nominal_features,
            feature_extractor=feature_extractor, summaries=patient_summaries, y=y
        )

        # HGBC
        #hgbc_train_score, hgbc_test_scores = hgbc_txt_emb(
        #    feature_extractor=feature_extractor, summaries=patient_summaries, y=y
        #)

        # Gather results
        labels.extend([
            f"{model_name} \n+ Logistic Regression",
        #    f"{model_name} \n+ HGBC"
        ])
        train_scores.extend([lr_train_score])#, hgbc_train_score])
        test_score_medians.extend([
            np.median(lr_test_scores),
            #np.median(hgbc_test_scores)
        ])
        test_score_mins.extend([
            np.min(lr_test_scores),
            #np.min(hgbc_test_scores)
        ])
        test_score_maxs.extend([
            np.max(lr_test_scores),
            #np.max(hgbc_test_scores)
        ])

    # Convert to arrays
    train_scores = np.array(train_scores)
    test_score_medians = np.array(test_score_medians)
    test_score_mins = np.array(test_score_mins)
    test_score_maxs = np.array(test_score_maxs)

    plot_bar_chart(
        filename="cl_concatenated",
        labels=labels,
        train_scores=train_scores,
        test_score_medians=test_score_medians,
        test_score_mins=test_score_mins,
        test_score_maxs=test_score_maxs
    )
