import pandas as pd


def get_data(task_name):
    # 353150 training samples
    df_train = pd.read_csv("../mimic/train.csv")

    # 88287 test samples
    df_test = pd.read_csv("../mimic/test.csv")

    features = [
        "age", "gender",

        "n_ed_30d", "n_ed_90d", "n_ed_365d",
        "n_hosp_30d", "n_hosp_90d", "n_hosp_365d",
        "n_icu_30d", "n_icu_90d", "n_icu_365d",

        "triage_temperature", "triage_heartrate", "triage_resprate",
        "triage_o2sat", "triage_sbp", "triage_dbp", "triage_pain", "triage_acuity",

        "chiefcom_chest_pain", "chiefcom_abdominal_pain", "chiefcom_headache",
        "chiefcom_shortness_of_breath", "chiefcom_back_pain", "chiefcom_cough",
        "chiefcom_nausea_vomiting", "chiefcom_fever_chills", "chiefcom_syncope",
        "chiefcom_dizziness",

        "cci_MI", "cci_CHF", "cci_PVD", "cci_Stroke", "cci_Dementia",
        "cci_Pulmonary", "cci_Rheumatic", "cci_PUD", "cci_Liver1", "cci_DM1",
        "cci_DM2", "cci_Paralysis", "cci_Renal", "cci_Cancer1", "cci_Liver2",
        "cci_Cancer2", "cci_HIV",

        "eci_Arrhythmia", "eci_Valvular", "eci_PHTN", "eci_HTN1", "eci_HTN2",
        "eci_NeuroOther", "eci_Hypothyroid", "eci_Lymphoma", "eci_Coagulopathy",
        "eci_Obesity", "eci_WeightLoss", "eci_FluidsLytes", "eci_BloodLoss",
        "eci_Anemia", "eci_Alcohol", "eci_Drugs", "eci_Psychoses", "eci_Depression"
    ]

    if task_name == "task_0":
        outcome = "outcome_inhospital_mortality"
    elif task_name == "task_1":
        outcome = "outcome_icu_transfer_12h"
    elif task_name == "task_2":
        outcome = "outcome_critical"
    elif task_name == "task_3":
        outcome = "outcome_hospitalization"

    X_train = df_train[features].copy()
    X_test = df_test[features].copy()
    y_labels_train = df_train[outcome].copy()
    y_labels_test = df_test[outcome].copy()

    return X_train, X_test, y_labels_train, y_labels_test


import pandas as pd


import pandas as pd

def is_binary(series):
    """Check if a pandas Series contains only 0 and 1 (ignores NaNs)."""
    return set(series.dropna().unique()) <= {0, 1}


def convert_and_save(train_path, test_path, out_train_path, out_test_path):
    # Load data
    X_train = pd.read_csv(train_path)
    X_test = pd.read_csv(test_path)

    # --- Process columns starting with 'chiefcom'
    for col in X_train.columns[X_train.columns.str.startswith("chiefcom")]:
        if col in X_test.columns:
            if is_binary(X_train[col]) and is_binary(X_test[col]):
                X_train[col] = X_train[col].astype(bool)
                X_test[col] = X_test[col].astype(bool)
            else:
                X_train[col] = X_train[col].astype(int).astype("category")
                X_test[col] = X_test[col].astype(int).astype("category")

    # --- Process 'cci' and 'eci' columns
    for category in ["cci", "eci"]:
        for col in X_train.columns[X_train.columns.str.startswith(category)]:
            if col in X_test.columns:
                if is_binary(X_train[col]) and is_binary(X_test[col]):
                    X_train[col] = X_train[col].astype(bool)
                    X_test[col] = X_test[col].astype(bool)
                else:
                    X_train[col] = X_train[col].astype("category")
                    X_test[col] = X_test[col].astype("category")

    # Save back to CSV
    X_train.to_csv(out_train_path, index=False)
    X_test.to_csv(out_test_path, index=False)


def get_cat_features(X_train, X_test):
    X_train["gender"] = X_train["gender"].astype("category")
    X_test["gender"] = X_test["gender"].astype("category")
    for col in X_train.columns[X_train.columns.str.startswith("chiefcom")]:
        X_train[col] = X_train[col].astype("int").astype("category")
        X_test[col] = X_test[col].astype("int").astype("category")
    for category in ["cci", "eci"]:
        for col in X_train.columns[X_train.columns.str.startswith(category)]:
            X_train[col] = X_train[col].astype("category")
            X_test[col] = X_test[col].astype("category")


"""
# Falls Subsampling gebraucht:
from imblearn.under_sampling import RandomUnderSampler        

if task_name == "task_0":
    outcome = "outcome_inhospital_mortality"
    sampling_strategy = "majority"
elif task_name == "task_1":
    outcome = "outcome_icu_transfer_12h"
    sampling_strategy = "majority"
elif task_name == "task_2":
    outcome = "outcome_critical"
    sampling_strategy = "majority"
elif task_name == "task_3":
    outcome = "outcome_hospitalization"
    sampling_strategy = {0: 5000, 1: 5000}
# resampling
rus = RandomUnderSampler(
    sampling_strategy=sampling_strategy, random_state=42)
X_train, y_labels_train = rus.fit_resample(X_train, y_labels_train)"""
