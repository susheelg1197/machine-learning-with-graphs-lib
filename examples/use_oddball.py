from ml_wg.OddBall_Algorithm.oddball import oddball, compute_egonet_features
import networkx as nx
import numpy as np

def main():
    # Create a graph with potential anomalies
    G = nx.gnp_random_graph(20, 0.1, seed=42)
    for u, v in G.edges():
        G.edges[u, v]['weight'] = np.random.rand()

    # Detect anomalies using OddBall algorithm
    anomalies = oddball(G)

    # Output the anomalies detected
    print("Anomalies detected:", anomalies)

if __name__ == "__main__":
    main()
