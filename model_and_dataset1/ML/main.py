import numpy as np
import pickle
import pandas as pd
import argparse
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
)
import os


def model_select(model):
    if model == "RF":
        return RandomForestClassifier(criterion="entropy", n_estimators=150)
    elif model == "AdaBoost":
        return AdaBoostClassifier(learning_rate=1.0, n_estimators=100)
    elif model == "LR":
        return LogisticRegression(max_iter=5000,penalty="l2")
    elif model == "SGD":
        return SGDClassifier(loss="log_loss", penalty="l1")
    elif model == "KNN":
        return KNeighborsClassifier(n_neighbors=6, p=2, weights="distance")
    elif model == "XGBoost":
        return XGBClassifier(n_estimators=200,
                                max_depth=6,
                                learning_rate=0.1,
                                colsample_bytree=0.8,
                                subsample=0.8,
                                gamma=0,
                                reg_alpha=0.1,
                                reg_lambda=1,
                                scale_pos_weight=1)
    else:
        return None



# 定义函数用于加载数据和构造特征
def load_and_process_data(file_path, nps_dic, protein_dic, columns=None):
    """加载数据并构造特征矩阵"""
    data = pd.read_csv(file_path, header=None)
    if columns is None:
        columns = [0, 2, 4]
    selected_data = data.loc[:, columns]

    nps = selected_data[columns[0]].values
    proteins = selected_data[columns[1]].values
    labels = selected_data[columns[2]].values

    # 矢量化构造特征矩阵
    features = np.array([
        np.concatenate((nps_dic[np_id], protein_dic[prot_id]))
        for np_id, prot_id in zip(nps, proteins)
    ])

    return features, labels


def cross_validate_model(nps_feature, proteins_feature, n_folds, model_cls, output_dir="./results"):
    """
    Perform cross-validation for a given model and save the results.

    Parameters:
        nps_feature (str): Feature type for natural products (e.g., "PubChem").
        proteins_feature (str): Feature type for proteins (e.g., "DPC").
        n_folds (int): Number of cross-validation folds.
        model_cls (str): Machine learning model (e.g., RF).
        output_dir (str): Directory to save the results.

    Returns:
        pd.DataFrame: DataFrame containing the evaluation metrics for each fold.
    """
    # Load features
    with open(f"../data/nps_feature/drug_{nps_feature}.pkl", 'rb') as file:
        nps_dic = pickle.load(file)
    with open(f"../data/protein_feature/protein_{proteins_feature}.pkl", 'rb') as file:
        protein_dic = pickle.load(file)

    results = []

    for fold in range(1, n_folds + 1):
        train_path = f"./dataset/fold_{fold}/train_data.csv"
        test_path = f"./dataset/fold_{fold}/test_data.csv"

        # Load and process data
        X_train, y_train = load_and_process_data(train_path, nps_dic, protein_dic)
        X_test, y_test = load_and_process_data(test_path, nps_dic, protein_dic)

        # Data Standardization
        scaler = StandardScaler()  # Initialize the scaler
        X_train = scaler.fit_transform(X_train)  # Fit on training data and transform
        X_test = scaler.transform(X_test)  # Transform test data using the same scaler

        # Train model
        model = model_select(model_cls)
        model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp)

        results.append({
            "Fold": fold,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "AUC": roc_auc,
            "Specificity": specificity,
        })

        print(f"Fold {fold} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC: {roc_auc:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print("************************")
    # Calculate mean of each metric
    mean_results = {
        "Accuracy": np.mean([result["Accuracy"] for result in results]),
        "Precision": np.mean([result["Precision"] for result in results]),
        "Recall": np.mean([result["Recall"] for result in results]),
        "F1 Score": np.mean([result["F1 Score"] for result in results]),
        "AUC": np.mean([result["AUC"] for result in results]),
        "Specificity": np.mean([result["Specificity"] for result in results]),
    }

    # Add mean to the results
    results.append({
        "Fold": "Mean",
        "Accuracy": mean_results["Accuracy"],
        "Precision": mean_results["Precision"],
        "Recall": mean_results["Recall"],
        "F1 Score": mean_results["F1 Score"],
        "AUC": mean_results["AUC"],
        "Specificity": mean_results["Specificity"],
    })

    # Save results to a CSV file
    results_df = pd.DataFrame(results)
    result_path = os.path.join(output_dir, f"{model_cls}_{nps_feature}_{proteins_feature}_{n_folds}_new.csv")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    results_df.to_csv(result_path, index=False)

    print(f"Cross-validation results saved to {result_path}")
    return results_df


if __name__ == "__main__":
    # Define valid options
    valid_nps_features = ["ECFP", "ASP", "PubChem"]
    valid_proteins_features = ["DPC", "CTriad", "AAC"]
    valid_models = ["RF", "XGBoost", "AdaBoost", "LR", "SGD", "KNN"]

    # Argument parser
    parser = argparse.ArgumentParser(description="Run cross-validation for a specified model.")
    parser.add_argument("-nps_feature", required=True, choices=valid_nps_features,
                        help=f"Feature type for natural products. Options: {', '.join(valid_nps_features)}")
    parser.add_argument("-proteins_feature", required=True, choices=valid_proteins_features,
                        help=f"Feature type for proteins. Options: {', '.join(valid_proteins_features)}")
    parser.add_argument("-model", required=True, choices=valid_models,
                        help=f"Model to use for training. Options: {', '.join(valid_models)}")
    parser.add_argument("-n_folds", type=int, default=5, help="Number of folds for cross-validation (default: 5)")

    args = parser.parse_args()


    # Run cross-validation
    results = cross_validate_model(args.nps_feature, args.proteins_feature, args.n_folds, args.model)