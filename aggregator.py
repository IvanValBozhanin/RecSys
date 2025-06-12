import pandas as pd
import glob

# Step 1: Load all CSV files
csv_files = glob.glob("sparsification_results_run*.csv")  # Make sure filenames follow pattern: results_1.csv, ..., results_5.csv
dataframes = []

for file in csv_files:
    df = pd.read_csv(file)
    df["source_file"] = file  # Optional: Track source
    dataframes.append(df)

# Step 2: Concatenate into one big DataFrame
all_data = pd.concat(dataframes, ignore_index=True)

# Step 3: Fill missing values in tau and p for consistent grouping
all_data['tau'] = all_data['tau'].fillna('')  # Keep as string
all_data['p'] = all_data['p'].fillna('')

# Step 4: Group by technique + parameters
grouped = all_data.groupby(['cov_type', 'tau', 'p'])

# Step 5: Aggregate
agg = grouped.agg(
    rmse_mean=('test_rmse', 'mean'),
    rmse_std=('test_rmse', 'std'),
    sparsity_mean=('sparsity', 'mean'),
    sparsity_std=('sparsity', 'std'),
    time_mean=('training_time', 'mean'),
    time_std=('training_time', 'std')
).reset_index()

# Step 6: Save results
agg.to_csv("aggregated_results.csv", index=False)
print("Aggregated results saved to 'aggregated_results.csv'")
print(agg.head())
