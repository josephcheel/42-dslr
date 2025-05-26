import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
import os
# Sample dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="This script is used to describe a dataset")
    parser.add_argument("dataset", type=str, help="The dataset to be used")

    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print(f"Error: The file '{args.dataset}' does not exist.")
        exit(1)
    df = pd.read_csv(args.dataset)
    temp = df.select_dtypes(include=['number'])
    # Compute correlation matrix
    correlation_matrix = temp.corr()

    # Extract unique correlation pairs
    corr_values = []
    feature_pairs = []

    correlation_matrix.drop_duplicates()
    corr_pairs = correlation_matrix.unstack().sort_values(ascending=False)
    corr_pairs = corr_pairs[corr_pairs < 1]

    # for i, txt in enumerate(corr_values):
    #     plt.annotate(f"{txt:.2f}", (feature_pairs[i], corr_values[i]), textcoords="offset points", xytext=(0,5), ha='center')
    
    sns.pairplot(correlation_matrix, plot_kws={'alpha': 0.6, 's': 80, 'edgecolor': 'k'}, height=2, palette=['#1f77b4', '#ff7f0e'])
    plt.tight_layout()
    plt.show()
