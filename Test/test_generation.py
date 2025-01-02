import numpy as np

def format_cpp_array(array, name="array"):
    rows = []
    for row in array:
        formatted_row = "{" + ", ".join(f"{x:.6f}" for x in row) + "}"
        rows.append(formatted_row)
    
    cpp_array = f"float {name}[][] = {{\n    " + ",\n    ".join(rows) + "\n};"
    return cpp_array

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
    return y_pred - y_true



def generate_cross_entropy_test(batch_size, num_classes):
    # Generate random inputs
    y_pred = random_predictions(batch_size, num_classes)
    y_true = random_labels(batch_size, num_classes)

    # Compute the gradient
    gradient = cross_entropy_backward(y_pred, y_true)

    # Print inputs and output
    print("Predictions (y_pred):\n", format_cpp_array(y_pred))
    print("Target labels (y_true):\n", format_cpp_array(y_true))
    print("Gradient (backward pass):\n", format_cpp_array(gradient))

def generate_backward_test(batch_size, in_features, out_features):
    weights = np.random.rand(in_features, out_features)
    gradient_next = np.random.rand(batch_size, out_features)

    output = gradient_next.dot(weights.T)

    print("Weights:\n", format_cpp_array(weights, name="weights"))
    print("Gradient (next):\n", format_cpp_array(gradient_next, name="gradient_next"))
    print("Output (dL/dX):\n", format_cpp_array(output, name="output"))
    

np.random.seed(42)
generate_backward_test(4, 3, 4);
