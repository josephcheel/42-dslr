import argparse
import os

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="This script is used to describe a dataset")
	parser.add_argument("dataset", type=str, help="The dataset to be used")

	args = parser.parse_args()
	
	if not os.path.exists(args.dataset):
		print(f"Error: The file '{args.dataset}' does not exist.")
		exit(1)
	df = pd.read_csv(args.dataset)
	
	print(df.head())
	temp = df.select_dtypes(include=['number'])

	
	data = df.dropna(axis=1, how = 'all')
	sns.pairplot(temp, plot_kws={'alpha': 0.6, 's': 80, 'edgecolor': 'k'}, height=2, hue='Index', palette=['#1f77b4', '#ff7f0e'])

	plt.tight_layout()
	plt.show()