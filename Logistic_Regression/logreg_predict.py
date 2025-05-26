import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import signal
import json
import pandas as pd
import os


signal.signal(signal.SIGINT, signal.SIG_DFL)

def ft_softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def predict(X, weights, biases):
	"""
	X: numpy array, shape (num_samples, num_features)
	weights: numpy array, shape (num_classes, num_features)
	biases: numpy array, shape (num_classes,)
	
	Returns:
	  - predicted class indices (list)
	  - probabilities (numpy array shape (num_samples, num_classes))
	"""
	try:
		logits = np.dot(X, weights) + biases  # shape (num_samples, num_classes)
		probs = ft_softmax(logits)  # shape (num_samples, num_classes)
		pred_indices = np.argmax(probs, axis=1)
	except ValueError as e:
		print(f"Error during prediction: {e}")
		print(f"X shape: {X.shape}, weights shape: {weights.shape}, biases shape: {biases.shape}")
		sys.exit(1)
	# Get predicted class indices for each sample
	
	return pred_indices.tolist(), probs

def load_model_parameters(input_path):
	try:
		with open(input_path, 'r') as f:
			data = json.load(f)
	except Exception as e:
		print(f"Error loading model parameters: {e}")
		sys.exit(1)

	try:
		classes = data["classes"]
		weights = np.array(data["weights"])
		biases = np.array(data["biases"])
		column_names = list(data["column_names"])
	except KeyError as e:
		print(f"Missing key in model parameters: {e}")
		sys.exit(1)
	except ValueError as e:
		print(f"Error converting model parameters to numpy arrays: {e}")
		sys.exit(1)
	except TypeError as e:
		print(f"Error with data types in model parameters: {e}")
		sys.exit(1)
	except Exception as e:
		print(f"Unexpected error: {e}")
		sys.exit(1)

	return classes, weights, biases, column_names

def isNumeric(value):
	try:
		float(value)
		return True
	except ValueError:
		return False

def load_dataset(dataset_path, delimiter=',', header=0, feature_names=None):
	try:
		df = pd.read_csv(dataset_path, delimiter=delimiter, skiprows=header)
		df = df.dropna(axis=1, how = 'all')
		df = df.dropna(axis = 0)
	except FileNotFoundError:
		print(f"File {dataset_path} not found. Please specify a valid path with option --dataset or -d", file=sys.stderr)
		exit(1)
	except PermissionError:
		print(f"Permission denied to access file {dataset_path}. Please check the permissions.", file=sys.stderr)
		exit(1)
	except Exception as e:
		print(f"An error occurred while parsing the dataset: {e}", file=sys.stderr)
		exit(1)
	
	try:
		X = df[feature_names].values
	except KeyError or ValueError as e:
		print(f"Error: {e}. Please check the feature names in the dataset.", file=sys.stderr)
		exit(1)
	except Exception as e:
		print(f"Unexpected error: {e}", file=sys.stderr)
		exit(1)

	return X

if __name__ == '__main__':
	argparser = argparse.ArgumentParser(
		description=""
	)
	
	# Dataset-related arguments
	dataset_group = argparser.add_argument_group('Dataset Options')
	dataset_group.add_argument('--dataset', "-d", type=str, required=True, help="Path to the test datasets to make the predictions (mandatory)")
	dataset_group.add_argument('--input', '-i', type=str, required=True, help='Path to the input JSON file containing the model parameters (mandatory)')
	dataset_group.add_argument('--output', "-o", type=str, default='result.csv', help="Output file for the results as CSV. (default: result.csv)")
	
	dataset_group.add_argument('--delimiter', "-del", type=str, default=',', help="Delimiter for the dataset.")
	dataset_group.add_argument('--skip_header', '-s', type=int, default=0, help="Skip header of the dataset. 0: no header, 1: skip first row, 2: skip first two rows")

	args = argparser.parse_args()

	classes, weights, biases, column_names = load_model_parameters(args.input)

	X = load_dataset(args.dataset, args.delimiter, args.skip_header, column_names)

	pred_indices, probs = predict(X, weights, biases)
	pred_classes = [classes[i] for i in pred_indices]
	
	result_df = pd.DataFrame({"Predicted Class": pred_classes})
	if not os.path.exists(args.output):
		print(f"Output file {args.output} does not exist. It will be created.")
	else:
		print(f"WARNING: Output file '{args.output}' already exists and will be OVERWRITTEN!")
		input = input("Press Enter to continue or Ctrl+C to cancel...")
	result_df.to_csv(args.output, index=True, index_label="Index")
	print(f"Results successfully saved to {args.output}")

