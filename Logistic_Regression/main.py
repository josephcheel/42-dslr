from logreg_train import ft_softmax 


softmax = ft_softmax([1.0, 2.0, 0.8])
print(softmax)

print(softmax.sum())

import json
import numpy as np

with open("model.json") as f:
    model = json.load(f)

classes = model["classes"]
weights = np.array(model["weights"])  # shape: (num_classes, num_features)
biases = np.array(model["biases"])    # shape: (num_classes,)

def softmax(z):
    # z shape: (num_classes, num_samples)
    e_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return e_z / e_z.sum(axis=0, keepdims=True)

def predict_batch(X):
    """
    X: numpy array, shape (num_samples, num_features)
    Returns:
      - predicted classes (list)
      - probabilities (numpy array shape (num_samples, num_classes))
    """
    # Compute logits: for each sample, get scores for each class
    # weights shape: (num_classes, num_features)
    # X.T shape: (num_features, num_samples)
    logits = np.dot(weights, X.T) + biases[:, np.newaxis]  # shape (num_classes, num_samples)

    probs = softmax(logits)  # shape (num_classes, num_samples)

    # Get predicted class indices for each sample
    pred_indices = np.argmax(probs, axis=0)

    # Map to class names
    pred_classes = [classes[i] for i in pred_indices]

    # Transpose probs to (num_samples, num_classes) for easier interpretation
    return pred_classes, probs.T

# Example usage: predict 3 samples at once
X_new = np.array([
    [1.0, 0.5, -0.2],
    [0.1, -0.4, 0.7],
    [0.3, 0.2, 0.0]
])

pred_classes, probabilities = predict_batch(X_new)

for i, (cls, prob) in enumerate(zip(pred_classes, probabilities)):
    print(f"Sample {i}: Predicted class = {cls}, Probabilities = {prob}")

