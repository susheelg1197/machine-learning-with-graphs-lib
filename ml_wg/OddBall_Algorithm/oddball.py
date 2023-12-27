import networkx as nx
import numpy as np
from scipy.stats import skew, kurtosis

def oddball(G):
    anomalies = []
    for node in G.nodes():
        egonet = nx.ego_graph(G, node)
        features = compute_egonet_features(egonet)
        
        # Anomaly detection based on statistical thresholds or a trained model
        if is_anomalous(features):
            anomalies.append(node)
    return anomalies
from scipy.stats import entropy

def compute_entropy(values):
    """
    Compute the entropy of a list of values.
    
    Args:
    - values (list): A list of numerical values.

    Returns:
    - float: The entropy of the values.
    """
    # Calculate the frequency of each value
    value_counts = np.bincount(values)
    probabilities = value_counts / np.sum(value_counts)

    # Compute entropy using scipy's entropy function
    return entropy(probabilities)
def compute_egonet_features(egonet):
    degrees = [deg for _, deg in egonet.degree()]
    
    # Check if the graph is weighted
    if 'weight' in nx.get_edge_attributes(egonet, 'weight'):
        edge_weights = [data.get('weight', 1) for _, _, data in egonet.edges(data=True)]
    else:
        edge_weights = [1] * egonet.number_of_edges()

    # Compute clustering coefficient
    clustering_coeffs = nx.clustering(egonet)
    average_clustering_coefficient = sum(clustering_coeffs.values()) / len(clustering_coeffs)

    # Compute additional features
    max_degree = max(degrees) if degrees else 0
    min_weight = min(edge_weights) if edge_weights else 0
    degree_entropy = compute_entropy(degrees)
    weight_entropy = compute_entropy(edge_weights)

    # Compile features
    features = {
        'average_degree': np.mean(degrees),
        'skewness_degree': skew(degrees),
        'kurtosis_degree': kurtosis(degrees),
        'average_weight': np.mean(edge_weights),
        'skewness_weight': skew(edge_weights),
        'kurtosis_weight': kurtosis(edge_weights),
        'edge_count': egonet.number_of_edges(),
        'max_degree': max_degree,
        'min_weight': min_weight,
        'degree_entropy': degree_entropy,
        'weight_entropy': weight_entropy,
        'clustering_coefficient': average_clustering_coefficient
    }
    return features


def is_anomalous(features):
    """
    Heuristic-based anomaly detection.

    Args:
    - features (dict): A dictionary of computed egonet features.

    Returns:
    - bool: True if the egonet is considered anomalous, False otherwise.
    """
    # Define thresholds for anomalies
    threshold_edge_count = 10
    threshold_clustering_coefficient = 0.2
    threshold_degree_entropy = 1.5
    threshold_weight_entropy = 1.5

    # Anomaly detection logic
    if features['edge_count'] > threshold_edge_count and features['clustering_coefficient'] < threshold_clustering_coefficient:
        return True
    if features['degree_entropy'] > threshold_degree_entropy or features['weight_entropy'] > threshold_weight_entropy:
        return True
    if features['max_degree'] / features['average_degree'] > 2 and features['min_weight'] < 0.1:
        return True

    # No anomalies detected
    return False

