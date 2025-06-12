import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Load data
output_file = (sys.argv[1] if len(sys.argv) > 1 else "graph") + ".png"
csv_file =    (sys.argv[1] if len(sys.argv) > 1 else "sparsification_results") + ".csv"
is_aggregated = (sys.argv[2] if len(sys.argv) > 2 else "False").lower() == "true"
df = pd.read_csv(csv_file)

# Clean and prepare
df.replace("", pd.NA, inplace=True)
df["tau"] = pd.to_numeric(df["tau"], errors="coerce")
df["p"] = pd.to_numeric(df["p"], errors="coerce")

if not is_aggregated:
    # Label for scatter
    def label(row):
        if pd.notna(row["p"]):
            return f"{row['cov_type']} (p={row['p']})"
        elif pd.notna(row["tau"]):
            return f"{row['cov_type']} (tau={row['tau']})"
        else:
            return row["cov_type"]

    df["label"] = df.apply(label, axis=1)

    # Normalize sparsity for point size
    df["size"] = (df["sparsity"] / df["sparsity"].max()) * 400 + 20  # size scaling

    # Plot
    plt.figure(figsize=(16, 10))
    scatter = plt.scatter(
        df["test_rmse"],
        df["training_time"],
        s=df["size"],
        c=df["sparsity"],
        cmap="viridis",
        alpha=0.8,
        edgecolor="k"
    )

    # Annotate points
    for i, row in df.iterrows():
        plt.annotate(row["label"], (row["test_rmse"], row["training_time"]),
                     fontsize=8, alpha=0.7, xytext=(3,3), textcoords='offset points')

    # Axes and title
    plt.xlabel("Test RMSE (↓ better)")
    plt.ylabel("Training Time (s)")
    plt.title("Trade-off between Accuracy, Time, and Sparsity")
    cbar = plt.colorbar(scatter)
    cbar.set_label("Sparsity (%)")
else:
    # For aggregated data with means and standard deviations
    def label(row):
        if pd.notna(row["p"]):
            return f"{row['cov_type']} (p={row['p']})"
        elif pd.notna(row["tau"]):
            return f"{row['cov_type']} (tau={row['tau']})"
        else:
            return row["cov_type"]

    df["label"] = df.apply(label, axis=1)

    # Normalize sparsity for point size
    df["size"] = (df["sparsity_mean"] / df["sparsity_mean"].max()) * 400 + 20

    # Group by covariance type for colors
    cov_types = df["cov_type"].unique()
    colors = plt.cm.tab10(range(len(cov_types)))
    cov_type_color_map = dict(zip(cov_types, colors))

    plt.figure(figsize=(16, 10))
    
    # Create scatter plot without error bars
    for cov_type in cov_types:
        cov_df = df[df["cov_type"] == cov_type]
        plt.scatter(
            cov_df["rmse_mean"], 
            cov_df["time_mean"],
            label=cov_type,
            s=cov_df["size"],
            color=cov_type_color_map[cov_type],
            alpha=0.8,
            edgecolor='k'
        )
    
    # Annotate points
    for i, row in df.iterrows():
        plt.annotate(
            row["label"], 
            (row["rmse_mean"], row["time_mean"]),
            fontsize=8, 
            alpha=0.7, 
            xytext=(3,3), 
            textcoords='offset points'
        )
    
    plt.legend()
    plt.xlabel("Test RMSE Mean (↓ better)")
    plt.ylabel("Training Time Mean (s)")
    plt.title("Trade-off between Accuracy, Time, and Sparsity (Aggregated Data)")
    
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    
    # Create a second plot showing sparsity like in the non-aggregated version
    plt.figure(figsize=(16, 10))
    scatter = plt.scatter(
        df["rmse_mean"],
        df["time_mean"],
        s=df["size"],
        c=df["sparsity_mean"],
        cmap="viridis",
        alpha=0.8,
        edgecolor="k"
    )
    
    # Annotate points
    for i, row in df.iterrows():
        plt.annotate(row["label"], (row["rmse_mean"], row["time_mean"]),
                     fontsize=8, alpha=0.7, xytext=(3,3), textcoords='offset points')
    
    # Axes and title
    plt.xlabel("Test RMSE Mean (↓ better)")
    plt.ylabel("Training Time Mean (s)")
    plt.title("Trade-off between Accuracy, Time, and Sparsity - Visualizing Sparsity")
    cbar = plt.colorbar(scatter)
    cbar.set_label("Mean Sparsity (%)")
    
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file.replace(".png", "_sparsity.png"), dpi=300)