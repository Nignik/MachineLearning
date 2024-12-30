import numpy as np

# Generate random predictions (y_pred) as probabilities
def random_predictions(batch_size, num_classes):
    logits = np.random.rand(batch_size, num_classes)  # Random logits
    y_pred = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)  # Softmax
    return y_pred

# Generate random target labels (y_true)
def random_labels(batch_size, num_classes):
    y_true = np.zeros((batch_size, num_classes))
    indices = np.random.randint(0, num_classes, size=batch_size)
    y_true[np.arange(batch_size), indices] = 1  # One-hot encoding
    return y_true

# Backward cross-entropy gradient
def cross_entropy_backward(y_pred, y_true):
    return (y_pred - y_true) / y_pred.shape[0]

# Example usage
batch_size = 4
num_classes = 3

def generate_cross_entropy_test():
    # Generate random inputs
    y_pred = random_predictions(batch_size, num_classes)
    y_true = random_labels(batch_size, num_classes)

    # Compute the gradient
    gradient = cross_entropy_backward(y_pred, y_true)

    # Print inputs and output
    print("Predictions (y_pred):\n", str(y_pred).replace('[', '{').replace(']', '}'))
    print("Target labels (y_true):\n", str(y_true).replace('[', '{').replace(']', '}'))
    print("Gradient (backward pass):\n", str(gradient).replace('[', '{').replace(']', '}'))
    
generate_cross_entropy_test()
