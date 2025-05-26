import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
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

    print(corr_pairs)
    print("Most similar features:", corr_pairs.idxmax())
    print("Correlation:", corr_pairs.max())
    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            corr_values.append(correlation_matrix.iloc[i, j])
            feature_pairs.append(f"{correlation_matrix.index[i]} vs {correlation_matrix.columns[j]}")
            print(f"{correlation_matrix.index[i]} vs {correlation_matrix.columns[j]}: {correlation_matrix.iloc[i, j]}")

    # Scatter plot of correlation values
    plt.figure(figsize=(40, 10))

    plt.scatter(feature_pairs, corr_values, color='b', alpha=0.7)
    # plt.scatter(corr_values, feature_pairs, color='b', alpha=1)
    
    plt.axhline(y=0, color='gray', linestyle='dashed', linewidth=1)  # Zero correlation reference line
    plt.ylim(-1, 1)
    plt.ylabel("Correlation Value")

    plt.title("Scatter Plot of Feature Correlations")

    # for i, txt in enumerate(corr_values):
    #     plt.annotate(f"{txt:.2f}", (corr_values[i], feature_pairs[i]), textcoords="offset points", xytext=(5, 0), ha='left')

    # Annotate each point with its correlation value
    for i, txt in enumerate(corr_values):
        plt.annotate(f"{txt:.2f}", (feature_pairs[i], corr_values[i]), textcoords="offset points", xytext=(0,5), ha='center')
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.yticks(np.arange(-1, 1.1, 0.1), fontsize=8, ha='right')
    plt.tight_layout()
    plt.show()
