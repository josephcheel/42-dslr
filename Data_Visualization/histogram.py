import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
	parser = argparse.ArgumentParser( description = "This script is used to describe a dataset")
	parser.add_argument("dataset", type=str, help="The dataset to be used")

	args = parser.parse_args()
	
	if not os.path.exists(args.dataset):
		print(f"Error: The file '{args.dataset}' does not exist.")
		exit(1)
	df = pd.read_csv(args.dataset)
	numeric_columns = df.select_dtypes(include=['number']).columns

	print(numeric_columns)

	cv_values = {col: df[col].std() / abs(df[col].mean()) for col in numeric_columns}

	# Convert to DataFrame for plotting
	cv_df = pd.DataFrame.from_dict(cv_values, orient='index', columns=['CV'])

	# Sort values for better visualization
	cv_df = cv_df.sort_values(by='CV')
	# Plot bar chart
	plt.figure(figsize=(12, 8))
	plt.bar(cv_df.index, cv_df['CV'], color='skyblue', edgecolor='black', label='Coefficient of Variation', alpha=0.7)

	# Add value labels on top of each bar
	for index, value in enumerate(cv_df['CV']):
		plt.text(index, value + 0.01, f'{value:.2f}', ha='center', fontsize=9, color='black')

	# Add title and labels
	plt.title('Homogeneity of Hogwarts Courses (CV)')
	plt.xlabel('Courses')
	plt.ylabel('Coefficient of Variation (CV)')
	plt.xticks(rotation=80)

	# Add legend and information about CV
	plt.legend()
	plt.text(0.5, 0.7, 'Lower CV values indicate more homogeneity', 
			 transform=plt.gca().transAxes, fontsize=10, color='darkred', ha='center')

	plt.tight_layout() # adjust plot to fit labels
	
	# Show plot
	plt.show()