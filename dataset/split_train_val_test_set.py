import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
file_path = '../data/akebono_solar_combined_filtered.tsv'
df = pd.read_csv(file_path, sep='\t')

# Split into train+val and test sets
train_val, test = train_test_split(df, test_size=100000, random_state=42)

# Split train+val into train and validation sets
train, val = train_test_split(train_val, test_size=100000, random_state=42)

# Print the size of each sample
print(f"Train set size: {len(train)}")
print(f"Validation set size: {len(val)}")
print(f"Test set size: {len(test)}")

# Save datasets to separate files
train.to_csv('../data/train_v3.tsv', sep='\t', index=False)
val.to_csv('../data/validation_v3.tsv', sep='\t', index=False)
test.to_csv('../data/test_v3.tsv', sep='\t', index=False)

print("Datasets saved to separate tsv files in the 'data' directory.")

