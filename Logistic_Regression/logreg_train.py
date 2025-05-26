import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import signal
import pandas as pd
from sklearn.model_selection import train_test_split
from logreg_predict import predict, ft_softmax
LEARNING_RATE = 0.01
MINIMUM_STEP_SIZE = 0.0001
MAXIMUM_NUMBER_OF_STEPS = 1000
STARTING_THETA0 = 0
STARTING_THETA1 = 0

error_list = []

signal.signal(signal.SIGINT, signal.SIG_DFL)

def cross_entropy_loss():
   pass

def ft_gradient_descend(theta0, theta1, observed_y, observed_x, errors):

    for step in range(MAXIMUM_NUMBER_OF_STEPS):

        #  , error = mean_bias_error(observed_x, observed_y, theta0, theta1)
        if errors:
            error_list.append(np.mean(error**2))
    
        if abs(new_theta0 - theta0) < MINIMUM_STEP_SIZE and abs(new_theta1 - theta1) < MINIMUM_STEP_SIZE:
            print(f"Converged at step {step}")
            break

        theta0, theta1 = new_theta0, new_theta1

    return theta0, theta1

def standardization(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std, mean, std

# def ft_linear_regression(data, errors):
    
#     # Standardize the data
#     data, mean_x, std_x = standardization(data)
#     standarized_x = data[:, 0]
#     standarized_y = data[:, 1]

#     theta0, theta1 = ft_gradient_descend(STARTING_THETA0, STARTING_THETA1, standarized_y, standarized_x, errors)

#     # Reverse the standardization
#     theta1 = theta1 * (std_x[1] / std_x[0])
#     theta0 = theta0 * std_x[1] + mean_x[1] - theta1 * mean_x[0]
    
#     return theta0, theta1

def ft_logistic_regression(intercept, coefficients, x):
    ft_sigmoid(intercept, coefficients, x)

def ft_softmax_regression():
    pass

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

def output_result(theta0, theta1, output_file):
    try:
        open(output_file, "w").write(f"{{\"theta0\": {theta0}, \"theta1\": {theta1}}}")
        print(f"Results successfully saved to {output_file}")
    except PermissionError:
        print(f"Permission denied to write to file {output_file}. Please check the permissions.", file=sys.stderr)

# theta0 = intercept
# theta1 = slope

def initialize_terminal_arguments(argparser):
    dataset_group = argparser.add_argument_group('Dataset Options')
    dataset_group.add_argument(
        '--dataset', "-d",
        type=str, 
        required=True, 
        default='./data.csv', 
        help="Path to the dataset."
    )

    dataset_group.add_argument(
        '--features', '-f',
        type=str,
        help='Comma-separated list of feature column names to use.'
    )
    
    dataset_group.add_argument(
        '--features_file', '-fl',
        type=str,
        help='Path to a file containing feature names, one per line.'
    )
    
    dataset_group.add_argument(
        '--target',
        type=str,
        required=True,
        help='Name of the target column.'
    )
    
    dataset_group.add_argument(
        '--delimiter', "-del",
        type=str,
        default=',',
        help="Delimiter for the dataset."
    )

    dataset_group.add_argument(
        '--skiprows', '-s',
        type=int,
        default=0,
        help="Skip header of the dataset. 0: no header, 1: skip first row, 2: skip first two rows"
    )

    # Algorithm options
    algo_group = argparser.add_argument_group('Algorithm Options')
    algo_group.add_argument('--learning_rate', "-lr", type=float, default=0.01, help="Learning rate for the Gradient Descent algorithm.")

    # Output options
    output_group = argparser.add_argument_group('Output Options')
    output_group.add_argument('--output', "-o", type=str, default='model.json', help="Output file for the results as JSON. (default: model.json)")

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description="Computes a Linear Regression using the Gradient Descent Algorithm with the specified dataset."
    )
    initialize_terminal_arguments(argparser)
    # Dataset-related arguments
    
    # Parse arguments
    args = argparser.parse_args()
    if args.learning_rate:
        LEARNING_RATE = args.learning_rate

    if not args.features and not args.features_file:
        argparser.error("any feature specified please use --features or --features_list")
        
    example = [[1.9, 1.2, 0.7]]
    print(ft_softmax(example)) # calculate probabilistics
    df = parse_dataset(args.dataset, delimiter=args.delimiter, skiprows=args.skiprows)

    dataframe = df.drop('Hogwarts House', axis=1)
    target_variables = df['Hogwarts House']
  
    sys.exit(1) 

    
    if data is None or data.size == 0 or len(data.shape) < 2 or data.shape[1] < 2:
        print("Error: CSV file has an incorrect format. Possible issues include too many columns, too few rows, or mismatched data. Exiting...")
        exit(1)

    print(f"Original data shape: {data.shape}")
    

    numeric_data = np.array([[pd.to_numeric(value, errors='coerce') for value in row] for row in data], dtype=float)

    valid_rows = ~np.isnan(data).any(axis=0)
    cleaned_data = data[valid_rows]
    cleaned_data = cleaned_data[~np.isnan(cleaned_data).any(axis=1)]

    print(f"Cleaned data shape: {cleaned_data.shape}")
    if cleaned_data is None or cleaned_data.size == 0 or len(cleaned_data.shape) < 2 or cleaned_data.shape[1] < 2:
        print("Error: CSV file has an incorrect format. Possible issues include too many columns, too few rows, or mismatched data. Exiting...")
        exit(1)
    
    # print(cleaned_data.shape, len(cleaned_data.shape))
    original_x = cleaned_data[:, 0].copy()
    original_y = cleaned_data[:, 1].copy()

    theta0, theta1 = ft_linear_regression(cleaned_data, args.errors)

    print(f"Theta0: {theta0}, Theta1: {theta1}")
    if args.output:
        output_result(theta0, theta1, args.output)
    if args.errors:
        if error_list:
            print(f"Last Error: {error_list[-1]}")

    if args.graphical:
        linear_regression_window(original_x, original_y, theta0, theta1)

    if args.errors:
        error_window(error_list)

m = observed = 12


def ft_cost_function(theta0, theta1, observed_x, observed_y):
    result = - 1/m * np.sum(observed_y * np.log(ft_sigmoid(theta0, theta1, observed_x)) + (1 - observed_y) * np.log(1 - ft_sigmoid(theta0, theta1, observed_x)))
