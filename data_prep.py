import os
import pandas as pd
# from imblearn.under_sampling import RandomUnderSampler

from data_preps import create_general_summaries_

# CYBERSECURITY
"""
column_name_map = {
        "network_packet_size": "Network packet size",
        "protocol_type": "Protocol type",
        "login_attempts": "Login attempts",
        "session_duration": "Session duration",
        "encryption_used": "Encryption used",
        "ip_reputation_score": "IP reputation score",
        "failed_logins": "Failed logins",
        "browser_type": "Browser type",
        "unusual_time_access": "Unusual time access",
    }
"""
"""
categorial_values = {
    "unusual_time_access": {0: "no", 1: "yes"}
}
"""


def mimic_add_preprocessing(file, output):
    """
    Replaces True/False with yes/no and M/F with Male/Female in string-like columns.
    Returns a modified copy of the DataFrame.
    """
    df = pd.read_csv(file)

    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].map({True: "yes", False: "no"})
        elif df[col].dtype == object or pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].replace({"M": "Male", "F": "Female"})
    df.to_csv(output, index=False)


def is_binary(series):
    """Check if a pandas Series contains only 0 and 1 (ignores NaNs)."""
    return set(series.dropna().unique()) <= {0, 1}


def mimic_convert_binary_to_bool(train_path, test_path, out_train_path, out_test_path):
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


def mimic_get_cat_features(X_train, X_test):
    X_train["gender"] = X_train["gender"].astype("category")
    X_test["gender"] = X_test["gender"].astype("category")
    for col in X_train.columns[X_train.columns.str.startswith("chiefcom")]:
        X_train[col] = X_train[col].astype("int").astype("category")
        X_test[col] = X_test[col].astype("int").astype("category")
    for category in ["cci", "eci"]:
        for col in X_train.columns[X_train.columns.str.startswith(category)]:
            X_train[col] = X_train[col].astype("category")
            X_test[col] = X_test[col].astype("category")


def mimic_nom_features(X):
    nominal_features = []

    # Gender
    if "gender" in X.columns:
        nominal_features.append("gender")

    # Chief, CCI + ECI
    for category in ["chiefcom", "cci", "eci"]:
        for col in X.columns[X.columns.str.startswith(category)]:
            nominal_features.append(col)

    return nominal_features


"""
def mimic_subsample(X_train, y_labels_train, task_name):
    X_train = pd.read_csv(X_train)
    #X_test = pd.read_csv(X_test)
    y_labels_train = pd.read_csv(y_labels_train).squeeze()
    #y_labels_test = pd.read_csv(y_labels_test).squeeze()

    if task_name == "task_0":
        outcome = "outcome_inhospital_mortality"
        sampling_strategy = "majority"
    elif task_name == "task_1":
        outcome = "outcome_icu_transfer_12h"
        sampling_strategy = {0: 5000, 1: 5000}
        #sampling_strategy = "majority"
    elif task_name == "task_2":
        outcome = "outcome_critical"
        sampling_strategy = {0: 5000, 1: 5000}
        #sampling_strategy = "majority"
    elif task_name == "task_3":
        outcome = "outcome_hospitalization"
        sampling_strategy = {0: 5000, 1: 5000}
    # resampling
    rus = RandomUnderSampler(
        sampling_strategy=sampling_strategy, random_state=42)

    X_train, y_labels_train = rus.fit_resample(X_train, y_labels_train)
    #X_test, y_labels_test = rus.fit_resample(X_test, y_labels_test)

    X_train.to_csv(f"../mimic/subsampled/X_train_{task_name}.csv", index=False)
    y_labels_train.to_csv(f"../mimic/subsampled/y_train_{task_name}.csv", index=False)
    #X_test.to_csv(f"../mimic/subsampled/X_test_{task_name}.csv", index=False)
    #y_labels_test.to_csv(f"../mimic/subsampled/y_test_{task_name}.csv", index=False)
"""


# Generate ML Tasks
def mimic_generate_tasks(task_name):
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

    X_train.to_csv(f"../mimic/tasks/X_train_{task_name}.csv", index=False)
    X_test.to_csv(f"../mimic/tasks/X_test_{task_name}.csv", index=False)
    y_labels_train.to_csv(f"../mimic/tasks/y_train_{task_name}.csv", index=False)
    y_labels_test.to_csv(f"../mimic/tasks/y_test_{task_name}.csv", index=False)

    return X_train, X_test, y_labels_train, y_labels_test


def print_csv_file_lengths(folder_path):
    """
    Prints the name and number of rows of each .csv file in the given folder.
    """
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            try:
                df = pd.read_csv(file_path)
                print(f"{filename}: {len(df)} rows")
            except Exception as e:
                print(f"Failed to read {filename}: {e}")


def do_subsample():
    task = "task_0"
    #mimic_subsample(X_train=fr"..\mimic\tasks\X_train_{task}.csv",
    #                X_test=fr"..\mimic\tasks\X_test_{task}.csv",
    #                y_labels_train=fr"..\mimic\tasks\y_train_{task}.csv",
    #                y_labels_test=fr"..\mimic\tasks\y_test_{task}.csv",
    #                task_name=task)


def do_add_prep():
    task = "task_3"
    path_in = fr'..\mimic\subsampled\nominal'
    path_out = fr'..\mimic\summaries_prep'

    mimic_convert_binary_to_bool(
        train_path=os.path.join(path_in, fr"X_nom_train_{task}.csv"),
        test_path=os.path.join(path_in, fr"X_nom_test_{task}.csv"),
        out_train_path=os.path.join(path_out, fr"X_nom_train_{task}.csv"),
        out_test_path=os.path.join(path_out, fr"X_nom_test_{task}.csv"),
    )

    mimic_add_preprocessing(file=os.path.join(path_out, fr"X_nom_train_{task}.csv"),
                            output=os.path.join(path_out, fr"X_nom_train_{task}.csv"))

    mimic_add_preprocessing(file=os.path.join(path_out, fr"X_nom_test_{task}.csv"),
                            output=os.path.join(path_out, fr"X_nom_test_{task}.csv"))


def create_mimic_summaries():
    task = "task_3"
    path_in = fr'..\mimic\summaries_prep'
    path_out = fr'..\mimic\summaries'
    create_general_summaries_(tab_data=os.path.join(path_in, fr"X_nom_train_{task}.csv"),
                              output_file=os.path.join(path_out, fr"mimic_{task}_train_nom_summaries.txt")
                              )
    create_general_summaries_(tab_data=os.path.join(path_in, fr"X_nom_test_{task}.csv"),
                              output_file=os.path.join(path_out, fr"mimic_{task}_test_nom_summaries.txt")
                              )

