import csv
import math
from collections import Counter

class Node:
    """
    Represents a node in the decision tree.
    """
    def __init__(self, feature=None, is_leaf=False, decision=None):
        """
        Args:
            feature: The feature name this node splits on (None for leaf nodes)
            is_leaf: True if this is a leaf node (decision node)
            decision: The decision value if this is a leaf node
        """
        self.feature = feature
        self.is_leaf = is_leaf
        self.decision = decision
        self.children = {}  # Dictionary mapping feature values to child nodes
    
    def add_child(self, value, child_node):
        """Add a child node for a specific feature value."""
        self.children[value] = child_node
    
    def __str__(self, level=0):
        """String representation for printing the tree."""
        indent = "  " * level
        if self.is_leaf:
            return f"{indent}Decision: {self.decision}\n"
        else:
            result = f"{indent}Feature: {self.feature}\n"
            for value, child in self.children.items():
                result += f"{indent}  [{self.feature} = {value}]\n"
                result += child.__str__(level + 2)
            return result


def calculate_entropy(data):
    """
    Calculate the entropy of a dataset based on the Decision column.
    
    Args:
        data: List of dictionaries representing the dataset
    
    Returns:
        float: The entropy value
    """
    if not data:
        return 0
    
    # Count the occurrences of each decision class
    decisions = [row['Decision'] for row in data]
    decision_counts = Counter(decisions)
    total = len(data)
    
    # Calculate entropy
    entropy = 0
    for count in decision_counts.values():
        if count > 0:
            proportion = count / total
            entropy -= proportion * math.log2(proportion)
    
    return entropy


def calculate_information_gain(feature_name, data):
    """
    Calculate the information gain for a given feature.
    
    Args:
        feature_name: Name of the feature to calculate information gain for
        data: List of dictionaries representing the dataset
    
    Returns:
        float: The information gain value
    """
    if not data:
        return 0
    
    # Calculate the entropy of the entire dataset
    total_entropy = calculate_entropy(data)
    
    # Get all unique values for the feature
    feature_values = set(row[feature_name] for row in data)
    
    # Calculate weighted entropy for each subset
    total_samples = len(data)
    weighted_entropy = 0
    
    for value in feature_values:
        # Create subset where feature has this value
        subset = [row for row in data if row[feature_name] == value]
        subset_size = len(subset)
        
        # Calculate proportion and entropy of this subset
        proportion = subset_size / total_samples
        subset_entropy = calculate_entropy(subset)
        
        # Add to weighted entropy
        weighted_entropy += proportion * subset_entropy
    
    # Information gain = total entropy - weighted entropy
    information_gain = total_entropy - weighted_entropy
    
    return information_gain


def build_decision_tree(data, remaining_features=None):
    """
    Recursively build a decision tree using information gain.
    
    Args:
        data: List of dictionaries representing the dataset
        remaining_features: List of features that haven't been used yet
    
    Returns:
        Node: The root node of the decision tree
    """
    # Initialize remaining features on first call
    if remaining_features is None:
        # Get all feature names except 'Decision'
        remaining_features = [key for key in data[0].keys() if key != 'Decision']
    
    # Base case 1: If no data, return None
    if not data:
        return None
    
    # Base case 2: If all decisions are the same, create a leaf node
    decisions = [row['Decision'] for row in data]
    unique_decisions = set(decisions)
    
    if len(unique_decisions) == 1:
        # All decisions are the same - create leaf node
        return Node(is_leaf=True, decision=decisions[0])
    
    # Base case 3: If no remaining features, return most common decision
    if not remaining_features:
        # Return the most common decision as a leaf
        most_common = Counter(decisions).most_common(1)[0][0]
        return Node(is_leaf=True, decision=most_common)
    
    # Recursive case: Find the best feature to split on
    best_feature = None
    best_ig = -1
    
    for feature in remaining_features:
        ig = calculate_information_gain(feature, data)
        if ig > best_ig:
            best_ig = ig
            best_feature = feature
    
    # Create a node for the best feature
    node = Node(feature=best_feature)
    
    # Get all unique values for the best feature
    feature_values = set(row[best_feature] for row in data)
    
    # Create remaining features list (excluding the current feature)
    new_remaining_features = [f for f in remaining_features if f != best_feature]
    
    # Split data and recursively build subtrees
    for value in feature_values:
        # Create subset where feature has this value
        subset = [row for row in data if row[best_feature] == value]
        
        # Recursively build subtree for this subset
        child_node = build_decision_tree(subset, new_remaining_features)
        
        if child_node:
            node.add_child(value, child_node)
    
    return node


def predict(node, sample):
    """
    Make a prediction for a single sample using the decision tree.
    
    Args:
        node: Root node of the decision tree
        sample: Dictionary representing a single data point
    
    Returns:
        str: The predicted decision
    """
    # If we're at a leaf node, return the decision
    if node.is_leaf:
        return node.decision
    
    # Otherwise, follow the tree based on the feature value
    feature_value = sample[node.feature]
    
    if feature_value in node.children:
        return predict(node.children[feature_value], sample)
    else:
        # If feature value not seen in training, return None or a default
        return None


def load_data(filename):
    """
    Load data from a CSV file.
    
    Args:
        filename: Path to the CSV file
    
    Returns:
        list: List of dictionaries representing the dataset
    """
    data = []
    with open(filename, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            data.append(row)
    return data


# Example usage
if __name__ == "__main__":
    # Load the dataset
    data = load_data('decision_tree_dataset.csv')
    
    print(f"Dataset loaded: {len(data)} samples\n")
    
    # Calculate information gain for each feature
    features = ['Age', 'Credit_rating', 'Student']
    
    print("Information Gain Analysis:")
    print("-" * 50)
    
    for feature in features:
        ig = calculate_information_gain(feature, data)
        print(f"{feature:15} | Information Gain: {ig:.6f}")
    
    print("-" * 50)
    print()
    
    # Build the decision tree
    print("Building Decision Tree...")
    tree = build_decision_tree(data)
    
    print("\nDecision Tree Structure:")
    print("=" * 50)
    print(tree)
    
    # Test predictions on a few samples
    print("\nTesting Predictions:")
    print("-" * 50)
    
    test_samples = [
        {'Age': 'young', 'Credit_rating': 'fair', 'Student': 'TRUE'},
        {'Age': 'senior', 'Credit_rating': 'excellent', 'Student': 'FALSE'},
        {'Age': 'middle-aged', 'Credit_rating': 'fair', 'Student': 'TRUE'},
    ]
    
    for i, sample in enumerate(test_samples, 1):
        prediction = predict(tree, sample)
        print(f"Sample {i}: {sample}")
        print(f"Prediction: {prediction}\n")
    
    # Calculate accuracy on training data
    print("Accuracy on Training Data:")
    print("-" * 50)
    correct = 0
    for row in data:
        sample = {k: v for k, v in row.items() if k != 'Decision'}
        prediction = predict(tree, sample)
        if prediction == row['Decision']:
            correct += 1
    
    accuracy = (correct / len(data)) * 100
    print(f"Accuracy: {correct}/{len(data)} = {accuracy:.2f}%")