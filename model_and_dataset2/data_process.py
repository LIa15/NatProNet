import pandas as pd

# Read the CSV file, specifying that it does not contain a header (header=None)
df = pd.read_csv('data/dataset2.csv', header=None)

# Define a cutoff value to categorize the numeric column
cutoff_value = 5

# Select the 2nd, 4th, and 5th columns from the dataset (indexing starts from 0 in pandas)
# Use the .loc method to explicitly select specific columns
selected_columns = df.loc[:, [1, 3, 4]]

# Ensure the 4th column contains numeric values; any non-numeric values will be coerced to NaN
selected_columns.loc[:, 4] = pd.to_numeric(selected_columns[4], errors='coerce')

# Apply a binary transformation to the 4th column:
# Values less than the cutoff_value are set to 0; values greater than or equal to it are set to 1
selected_columns.loc[:, 4] = selected_columns[4].apply(lambda x: 0 if x < cutoff_value else 1)

# Save the processed data to a new text file, using a space as the delimiter
# The filename includes the cutoff_value to indicate the applied threshold
selected_columns.to_csv('data/dataset2_cutoff_{}.txt'.format(cutoff_value), sep=' ', index=False, header=False)

# Print a confirmation message indicating that the data has been successfully saved
print("Data has been saved to the file.")
