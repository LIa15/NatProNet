import subprocess

# Define the valid options
# valid_nps_features = ["ECFP", "ASP", "PubChem"]
valid_nps_features = ["PubChem"]
valid_proteins_features = ["DPC", "CTriad", "AAC"]
valid_models = ["SGD"]
n_folds_values = [5]  # Add more fold options if needed


# Generate all combinations of parameters
all_tasks = []
for model in valid_models:
    for nps_feature in valid_nps_features:
        for proteins_feature in valid_proteins_features:
            all_tasks.append((nps_feature, proteins_feature, model))

# Open the checkpoint file for writing updates
for task in all_tasks:

    # Build the command
    cmd = [
        "python", "main.py",
        "-nps_feature", task[0],
        "-proteins_feature", task[1],
        "-model", task[2],
        "-n_folds", str(n_folds_values[0])
    ]

    # Print the command for debugging
    print(f"Running: {' '.join(cmd)}")

    # Execute the command
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Output:\n{result.stdout}")

    except subprocess.CalledProcessError as e:
        print(f"Error while running: {' '.join(cmd)}")
        print(f"Error Output:\n{e.stderr}")
        break  # Stop further execution to debug the error
