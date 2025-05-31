#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import signal
import pandas as pd
from sklearn.model_selection import train_test_split
from logreg_predict import predict, ft_softmax
import json
from sklearn.metrics import precision_score
import os 
from enum import Enum, auto

LEARNING_RATE = None
MINIMUM_STEP_SIZE = None
MAXIMUM_NUMBER_OF_STEPS = None

signal.signal(signal.SIGINT, signal.SIG_DFL)

def cross_entropy_loss(probabilities, one_hot_labels):
    eps = 1e-8  # small value for stability
    log_probs = -np.log(probabilities + eps)
    loss = np.mean(np.sum(one_hot_labels * log_probs, axis=1))
    return loss

def standardization(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    standardized_data = (data - mean) / std
    return standardized_data, mean, std

class GradientDescentType(Enum):
    BATCH = auto()
    STOCHASTIC = auto()
    MINI_BATCH = auto()

def ft_softmax_regression(starting_weights, starting_bias, data, one_hot_labels, gradient_type=GradientDescentType.BATCH, batch_size=32, errors=False):
    data, mean_x, std_x = standardization(data)

    if gradient_type == GradientDescentType.BATCH:
        print("Using Batch Gradient Descent")
        weights, bias, loss = ft_gradient_descend(starting_weights, starting_bias, data, one_hot_labels, errors=errors)
    elif gradient_type == GradientDescentType.STOCHASTIC:
        print("Using Stochastic Gradient Descent")
        weights, bias, loss = ft_gradient_descend_stochastic(starting_weights, starting_bias, data, one_hot_labels, errors=errors)
    elif gradient_type == GradientDescentType.MINI_BATCH:
        print(f"Using mini-batch gradient descent with batch size {batch_size}")
        if batch_size <= 0:
            print("Error: Mini-batch size must be a positive integer.")
            sys.exit(1)
        if batch_size > data.shape[0]:
            print(f"Warning: Mini-batch size {batch_size} is larger than the number of samples {data.shape[0]}. Using full batch instead.")
            batch_size = data.shape[0]
        weights, bias, loss = ft_gradient_descend_mini_batch(starting_weights, starting_bias, data, one_hot_labels, batch_size=batch_size, errors=errors)
    weights = weights / std_x[:, np.newaxis]
    bias = bias - (weights.T @ mean_x)

    return weights, bias, loss

def ft_gradient_descend(weights, bias, X, one_hot_labels, errors=False):
    error_list = []
    prev_loss = float('inf')
    for step in range(MAXIMUM_NUMBER_OF_STEPS):
        logits = X @ weights + bias
        probabilities = ft_softmax(logits)
        loss = cross_entropy_loss(probabilities, one_hot_labels)

        grad_logits = (probabilities - one_hot_labels) / one_hot_labels.shape[0]
        grad_weights = X.T @ grad_logits
        grad_bias = np.sum(grad_logits, axis=0)

        # Gradient update
        weights -= LEARNING_RATE * grad_weights
        bias -= LEARNING_RATE * grad_bias

        if errors:
            error_list.append(loss)

        # Convergence check
        if abs(loss - prev_loss) < MINIMUM_STEP_SIZE:
            print(f"Converged at step {step}")
            break
        prev_loss = loss
    
    return weights, bias, error_list

def ft_gradient_descend_stochastic(weights, bias, X, one_hot_labels, errors=False):
    weights, bias, error_list = ft_gradient_descent_step(weights, bias, X, one_hot_labels, batch_size=1, errors=errors)
    return weights, bias, error_list

def ft_gradient_descend_mini_batch(weights, bias, X, one_hot_labels, batch_size=32, errors=False):
    weights, bias, error_list = ft_gradient_descent_step(weights, bias, X, one_hot_labels, batch_size=batch_size, errors=errors)
    return weights, bias, error_list

def ft_gradient_descent_step(weights, bias, X, one_hot_labels, batch_size=32,errors=False):
    error_list = []
    prev_loss = float('inf')
    num_samples = X.shape[0]

    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    X_shuffled = X[indices]
    y_shuffled = one_hot_labels[indices]

    for step in range(MAXIMUM_NUMBER_OF_STEPS):
        for batch_start in range(0, X_shuffled.shape[0], batch_size):
            batch_end = min(batch_start + batch_size, X_shuffled.shape[0])
            X_batch = X_shuffled[batch_start:batch_end]
            one_hot_labels_batch = y_shuffled[batch_start:batch_end]
            logits = X_batch @ weights + bias
            probabilities = ft_softmax(logits)
            loss = cross_entropy_loss(probabilities, one_hot_labels_batch)
            grad_logits = (probabilities - one_hot_labels_batch) / one_hot_labels_batch.shape[0]  # Shape: (N, C)
            grad_weights = X_batch.T @ grad_logits        # Shape: (C, D)
            grad_bias = np.sum(grad_logits, axis=0)  # Shape: (C,)
            # Convergence check
            if abs(loss - prev_loss) < MINIMUM_STEP_SIZE:
                # print(f"Converged at step {step}")
                break
            prev_loss = loss

            # Gradient update
            weights -= LEARNING_RATE * grad_weights
            bias -= LEARNING_RATE * grad_bias

            if errors:
                error_list.append(loss)


    return weights, bias, error_list

def compute_gradient(constants: np.ndarray, feature: np.ndarray, label: np.ndarray, length: int) -> np.ndarray:
    """
    Compute the gradient of the cost function for linear regression.

    Parameters:
    - constants (np.ndarray): The model coefficients (theta) of shape (n_features,).
    - feature (np.ndarray): The feature matrix of shape (m, n_features), where m is the number of training examples.
    - label (np.ndarray): The actual labels of shape (m,).
    - length (int): The number of training examples (m).

    Returns:
    - np.ndarray: The gradient of the cost function, which will have shape (n_features,).
    """
    predictions = np.dot(feature, constants)  # h_theta(x), predictions of shape (m,)
    error = predictions - label  # The error vector of shape (m,)

    gradient = (1 / length) * np.dot(feature.T, error)  # Gradient of shape (n_features,)
    
    return gradient


def parse_dataset(file_path, delimiter=',', skiprows=0):
    try:
        data = pd.read_csv(file_path, delimiter=delimiter, skiprows=skiprows)
    except FileNotFoundError:
        print(f"File {file_path} not found. Please specify a valid path with option --dataset or -d", file=sys.stderr)
        exit(1)
    except PermissionError:
        print(f"Permission denied to access file {file_path}. Please check the permissions.", file=sys.stderr)
        exit(1)
    except Exception as e:
        print(f"An error occurred while parsing the dataset: {e}", file=sys.stderr)
        exit(1)
    return data

def output_result(output_file, weights, bias, classes, column_names):
    try:
        if output_file.endswith('.json'):
            output_file = output_file
        else:
            output_file += '.json'
        if os.path.exists(output_file):
            print(f"Warning: Output file {output_file} already exists. It will be overwritten.", file=sys.stderr)
            input("Press Enter to continue or Ctrl+C to cancel...")        
        open(output_file, "w").write(
            json.dumps({
                "classes": classes,
                "weights": weights.tolist(),
                "biases": bias.tolist(),
                "column_names": column_names
            }, indent=4)  # Use indent for pretty printing
        )
        print(f"Results successfully saved to {output_file}")
    except PermissionError:
        print(f"Permission denied to write to file {output_file}. Please check the permissions.", file=sys.stderr)


def initialize_terminal_arguments(argparser):
    dataset_group = argparser.add_argument_group('Dataset Options')
    dataset_group.add_argument('--dataset', "-d", type=str, required=True, default='./data.csv', help="Path to the dataset.")
    dataset_group.add_argument('--target', '-t', type=str, required=True, help='Name of the target column.')
    dataset_group.add_argument('--delimiter', "-del", type=str, default=',', help="Delimiter for the dataset.")
    dataset_group.add_argument('--skiprows', '-s', type=int, default=0, help="Skip header of the dataset. 0: no header, 1: skip first row, 2: skip first two rows")
    features_group = dataset_group.add_mutually_exclusive_group(required=True)
    features_group.add_argument('--features', '-f', nargs='+', type=str, help='List of feature names to use for training, space separated. Example: --features feature1 feature2 feature3')
    features_group.add_argument('--features_file', '-fl', type=str, help='Path to a file containing feature names, one per line.')
    
    # Algorithm options
    algo_group = argparser.add_argument_group('Algorithm Options')
    algo_group.add_argument('--learning_rate', "-lr", type=float, default=0.01, help="Learning rate for the Gradient Descent algorithm.")
    algo_group.add_argument('--max_steps', "-ms", type=int, default=1000, help="Maximum number of steps for the Gradient Descent algorithm.")
    algo_group.add_argument('--min_step_size', "-mss", type=float, default=0.0001, help="Minimum step size for convergence in the Gradient Descent algorithm.")
    
    optimization_group = algo_group.add_mutually_exclusive_group(required=False)
    optimization_group.add_argument('--batch', '-b', action='store_true', help="Use Batch Gradient Descent (default).")
    optimization_group.add_argument('--stochastic', '-st', action='store_true', help="Use Stochastic Gradient Descent instead of Batch Gradient Descent.")
    optimization_group.add_argument('--mini_batch', '-mb', nargs='?', type=int, default=None, const=32, help="Batch size for Stochastic Gradient Descent. (default: 32)")
    # algo_group.add_argument('--stochastic', '-st', action='store_true', help="Use Stochastic Gradient Descent instead of Batch Gradient Descent(default).")
    # algo_group.add_argument('--mini_batch', '-mb', nargs='?', type=int, default=None, const=32, help="Batch size for Stochastic Gradient Descent. (default: 32)", )
    algo_group.add_argument('--validation_split', '-vs', type=float, choices=[0.2, 0.3, 0.4], help="Fraction of the dataset to use for validation. (default: 0.2)")

    # Output options
    output_group = argparser.add_argument_group('Output Options')
    output_group.add_argument('--output', "-o", type=str, default='model.json', help="Output file for the results as JSON. (default: model.json)")
    output_group.add_argument('--errors', '-e', action='store_true', help="Enable error logging during training.")

def clean_dataframe_and_extract_info(df, validation_split=None):
    DROP_TYPE = 0
    
    try:
        match DROP_TYPE:
            case 0:
                df_tmp = df.dropna()
            case 1:
                # Drop columns with any NaN values
                df_tmp = df.fillna(df.mean(numeric_only=True))
            case 2:
                # Fill NaN values with column mean
                df_tmp = df.fillna(0)
            case _:
                df_tmp = df.dropna()
    except Exception as e:
        print(f"An error occurred while cleaning the dataset: {e}", file=sys.stderr)
        sys.exit(1)
    
    # if validation_split is not None:
    #     df_train, df_val = train_test_split(df, test_size=validation_split, random_state=42)

    try:
        df_cleaned = df_tmp.drop(args.target, axis=1)
        target_variables = df_tmp[args.target]
        unique_targets_sorted = sorted(target_variables.unique())
    except KeyError:
        print(f"Error: The target column '{args.target}' does not exist in the dataset. Please check your dataset and target column name.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while processing the dataset: {e}", file=sys.stderr)
        sys.exit(1)
    
    if len(unique_targets_sorted) < 2:
        print(f"Error: No more than one unique class target found. Please check your target column {args.target}.", file=sys.stderr)
        sys.exit(1)

    return df_cleaned, target_variables, unique_targets_sorted

def load_features_file(input_path):
    try:
        with open(input_path, 'r') as f:
            features = json.load(f)
    except Exception as e:
        print(f"Error loading model parameters: {e}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(features, list):
        print("Error: Model parameters should be a list of strings", file=sys.stderr)
        sys.exit(1)
    return features

def one_hot_encode(targets, unique_targets):
    """
    One-hot encode the target variable.
    
    Parameters:
    - targets: The target variable to encode.
    - unique_targets: The unique values in the target variable.
    
    Returns:
    - np.ndarray: One-hot encoded matrix.
    """
    one_hot = np.zeros((len(targets), len(unique_targets)))
    for i, target in enumerate(targets):
        one_hot[i, unique_targets.index(target)] = 1 # same as one_hot[i][unique_targets.index(target)] = 1 but more efficient
    return one_hot

def error_visualization(loss: list):
    plt.figure(figsize=(8, 5))
    plt.plot(loss, label='Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

def get_features(arg_features, features_file):
    if arg_features:
        features_names = arg_features
    elif features_file:
        features_names = load_features_file(features_file)
    if len(features_names) == 0:
        print("Error: No features specified. Please provide at least one feature using --features or --features_file.", file=sys.stderr)
        sys.exit(1)

    try:
        df_selected_features_values = df_cleaned[features_names]
    except KeyError as e:
        print(f"Error: One or more specified features do not exist in the dataset: {e}", file=sys.stderr)
        sys.exit(1)

    if df_selected_features_values.empty:
        print("Error: The dataset is empty after applying the specified features. Please check your dataset and feature selection.", file=sys.stderr)
        sys.exit(1)

    return features_names, df_selected_features_values

def precision_calculation(df_selected_features_values, weights, bias, one_hot):
    predicted_probs = ft_softmax(np.dot(df_selected_features_values, weights) + bias)
    predicted_labels = np.argmax(predicted_probs, axis=1)
    true_labels = np.argmax(one_hot, axis=1)

    precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    return precision

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description="Computes a Linear Regression using the Gradient Descent Algorithm with the specified dataset."
    )
    initialize_terminal_arguments(argparser)
    args = argparser.parse_args()

    LEARNING_RATE = args.learning_rate
    MINIMUM_STEP_SIZE = args.min_step_size
    MAXIMUM_NUMBER_OF_STEPS = args.max_steps

    df = parse_dataset(args.dataset, delimiter=args.delimiter, skiprows=args.skiprows)


    df_cleaned, target_variables, unique_targets_sorted = clean_dataframe_and_extract_info(df)

    features_names, df_selected_features_values = get_features(args.features, args.features_file)

    one_hot = one_hot_encode(target_variables, unique_targets_sorted)

    number_of_input_features = df_selected_features_values.shape[1]
    num_classes = len(unique_targets_sorted)
    starting_weights = np.random.randn(number_of_input_features, num_classes) * 0.01  # Initialize weights with small random values
    starting_bias =  np.zeros(num_classes)

    if args.batch:
        weights, bias, loss = ft_softmax_regression(
            starting_weights, starting_bias, df_selected_features_values.values, one_hot,
            gradient_type=GradientDescentType.BATCH, batch_size=None, errors=args.errors)
    elif args.stochastic:
        weights, bias, loss = ft_softmax_regression(
            starting_weights, starting_bias, df_selected_features_values.values, one_hot,
            gradient_type=GradientDescentType.STOCHASTIC, batch_size=1, errors=args.errors)
    elif args.mini_batch is not None:
        weights, bias, loss = ft_softmax_regression(
            starting_weights, starting_bias, df_selected_features_values.values, one_hot,
            gradient_type=GradientDescentType.MINI_BATCH,
            batch_size=args.mini_batch, errors=args.errors
        )
    else:
        weights, bias, loss = ft_softmax_regression(
            starting_weights, starting_bias, df_selected_features_values.values, one_hot,
            gradient_type=GradientDescentType.BATCH, batch_size=None, errors=args.errors)

    if args.errors:
        error_visualization(loss)
    
    precision = precision_calculation(df_selected_features_values, weights, bias, one_hot)
    print(f"Precision: {precision:.4f}")

    if args.output:
        output_result(output_file=args.output, weights=weights, bias=bias, classes=unique_targets_sorted, column_names=features_names)

