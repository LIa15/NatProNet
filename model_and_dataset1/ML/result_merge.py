import os
import pandas as pd


def read_mean_rows_from_csvs(directory):
    # List to store results
    header_name = ["model", "nps_feature", "protein_feature", "Accuracy", "Precision", "Recall", "F1 Score", "AUC", "Specificity"]
    data = []
    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_info = filename.split("_")[:3]
            file_path = os.path.join(directory, filename)
            try:
                # Read the CSV file
                df = pd.read_csv(file_path, index_col=0)

                last_row = (df.iloc[-1].values).tolist()
                data.append(file_info + last_row)
                print(last_row)
            except Exception as e:
                print(f"Error reading file {filename}: {e}")
    print(data)
    merge_df = pd.DataFrame(data=data, columns=header_name)
    merge_df.to_excel("merge.xlsx", index=False, sheet_name="Sheet1")


if __name__ == "__main__":
    directory = "./results/"
    results = read_mean_rows_from_csvs(directory)
