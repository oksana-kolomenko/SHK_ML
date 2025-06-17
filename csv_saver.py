import numpy as np
import pandas as pd


csv_names = [
    "no_emb_train_results",
    "no_emb_test_results",

    "rt_emb_train_results",
    "rt_emb_test_results",

    "txt_emb_train_results",
    "txt_emb_test_results"
    ]


def save_results_to_csv(output_file, dataset_name, ml_method, emb_method, concatenation,
                        best_params, pca_n_comp, metrics, is_train):
    """
    Save logistic regression results to a CSV file.

    Args:
        output_file (str): Path to the CSV file to save results.
        dataset_name (str): Name of the dataset.
        ml_method (str): Machine learning method used.
        emb_method (str): Embedding method used.
        metrics (dict or list): Metrics dictionary (for train) or list of dictionaries (for test).
        is_train (bool): Whether the data corresponds to training metrics. Default is True.

    Returns:
        None
    """
    data = []

    if is_train:
        # Training data: Single metrics dictionary
        metrics["Dataset"] = dataset_name
        metrics["ML Method"] = ml_method
        metrics["Embedding Method"] = emb_method
        metrics["Concatenation"] = concatenation
        metrics["Best Parameters"] = best_params
        metrics["PCA n_components"] = pca_n_comp
        metrics = {key: float(value) if isinstance(value, np.float64) else value for key, value in metrics.items()}
        data.append(metrics)
    else:
        # Test data: List of metrics dictionaries
        for fold_metrics in metrics:
            fold_metrics["Dataset"] = dataset_name
            fold_metrics["ML Method"] = ml_method
            fold_metrics["Embedding Method"] = emb_method
            fold_metrics["Concatenation"] = concatenation
            fold_metrics["Best Parameters"] = best_params
            fold_metrics["PCA n_components"] = pca_n_comp
            fold_metrics = {key: float(value) if isinstance(value, np.float64) else value for key, value in
                            fold_metrics.items()}

            data.append(fold_metrics)

    df = pd.DataFrame(data)

    if is_train:
        column_order = [
            "Dataset", "ML Method", "Embedding Method", "Concatenation", "Best Parameters", "PCA n_components",
            "AUC", "AP", "Sensitivity", "Specificity", "Precision", "F1", "Balanced Accuracy"
        ]
    else:
        column_order = [
            #"Fold",
            "Dataset", "ML Method", "Embedding Method", "Concatenation", "Best Parameters", "PCA n_components",
            "AUC", "AP", "Sensitivity", "Specificity", "Precision", "F1", "Balanced Accuracy"
        ]

    additional_columns = [col for col in df.columns if col not in column_order]
    full_column_order = column_order + additional_columns

    df = df[full_column_order]

    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
