import argparse
import os

import pandas as pd
import math

def get_percentils(percent, sorted_column, count):
	raw_position = (count - 1) * percent
	lowest_position = math.floor(raw_position)
	highest_position = math.ceil(raw_position)

	lower_value = sorted_column[lowest_position]
	upper_value = sorted_column[highest_position]
	decimal_part = raw_position - lowest_position
	
	return lower_value + (decimal_part * (upper_value - lower_value))

def describe(df: pd.core.frame.DataFrame) -> None:
	"""
	This function is used to describe a dataset

	Parameters:
		dataset (dict): The dataset to be described

	Returns:
		dict: The dictionary containing the description of the dataset
	"""
	describe_dict = {}
	for key in df.keys():
		if df[key].dtype != "float64" and df[key].dtype != "int64":
			continue

		clean_df = df[key].dropna()
		name = key
		count = len(clean_df)
		
		try:
			mean = clean_df.sum() / count
		except:
			mean = math.nan
		try:
			if count == 0:
				std = math.nan
			else:
				std = (((clean_df - mean)**2).sum() / (count - 1)) ** 0.5
		except:
			std = math.nan
		try:
			sorted_column = sorted(clean_df)
			min = sorted_column[0]
			max = sorted_column[count - 1]
			per25 = get_percentils(0.25, sorted_column, count)
			per50 = get_percentils(0.50, sorted_column, count)
			per75 = get_percentils(0.75, sorted_column, count)
		except:
			min = math.nan
			max = math.nan
			per25 = math.nan
			per50 = math.nan
			per75 = math.nan
		
		
		describe_dict[name] = {
			"Count": count,
			"Mean": mean,
			"Std": std,
			"Min": min,
			"25%": per25,
			"50%": per50,
			"75%": per75,
			"Max": max
		}
	return describe_dict


if __name__ == '__main__':
	parser = argparse.ArgumentParser( description = "This script is used to describe a dataset")
	parser.add_argument("dataset", type=str, help="The dataset to be used")

	args = parser.parse_args()
	
	if not os.path.exists(args.dataset):
		print(f"Error: The file '{args.dataset}' does not exist.")
		exit(1)
	try:
		df = pd.read_csv(args.dataset)
	except Exception as e:
		print(f"Error: {e}")
		exit(1)
	print(df.describe())
	dictionary_values = describe(df)

	df = pd.DataFrame(dictionary_values)
	print(df)


	
	
	