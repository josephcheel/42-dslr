import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="This script is used to describe a dataset")
	parser.add_argument("dataset", type=str, help="The dataset to be used")

	args = parser.parse_args()
	
	if not os.path.exists(args.dataset):
		print(f"Error: The file '{args.dataset}' does not exist.")
		exit(1)
	import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv(args.dataset)

# Select only numeric columns
numeric_cols = df.select_dtypes(include=['number']).columns

# Loop through all pairs of numeric columns
correlations = []

# Compute correlations and store them
for i in range(len(numeric_cols)):
	for j in range(i + 1, len(numeric_cols)):  # avoid repeating pairs
		col1 = numeric_cols[i]
		col2 = numeric_cols[j]

		# Drop NaN values for both columns
		df_clean = df[[col1, col2]].dropna()

		# Compute Pearson correlation
		corr = df_clean[col1].corr(df_clean[col2])

		# Store the correlation and column pair
		correlations.append((corr, col1, col2))

# Sort correlations by absolute value in descending order
correlations.sort(key=lambda x: x[0], reverse=False)

# Plot and print correlations in sorted order
for corr, col1, col2 in correlations:
	print(f"Pearson Correlation between {col1} and {col2}: {corr:.4f}")
	
	# Clean DataFrame by dropping NaN values for the current columns
	df_clean = df[[col1, col2]].dropna()
	
	# Assign colors: blue for col1, red for col2
	colors = ['blue'] * len(df_clean[col1])  # All points of col1 are blue
	colors += ['red'] * len(df_clean[col2])  # All points of col2 are red
	
	# Combine col1 and col2 data into one list
	data = pd.concat([df_clean[col1], df_clean[col2]], axis=0)
	
	# Set up the scatter plot size
	plt.figure(figsize=(7, 5))
	
	# Scatter plot with blue for col1 and red for col2
	sns.scatterplot(x=data.index, y=data.values, hue=colors, palette={'blue': 'blue', 'red': 'red'}, legend=False, alpha=0.7, edgecolor="black")
	
	# Add plot title and labels
	plt.title(f'Scatter Plot: {col1} vs {col2}\nCorrelation: {corr:.4f}')
	plt.xlabel(col1)
	plt.ylabel(col2)
	plt.grid(True)
	plt.show()