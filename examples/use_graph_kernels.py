import networkx as nx
from ml_wg.classification.graph_kernels import graphlet_kernel

def main():
    # Create two example graphs
    G1 = nx.gnp_random_graph(10, 0.5, seed=42)
    G2 = nx.gnp_random_graph(10, 0.5, seed=43)

    # Compute the Graphlet kernel between G1 and G2
    kernel_value = graphlet_kernel(G1, G2, k=3)
    print("Graphlet Kernel between G1 and G2:", kernel_value)

if __name__ == "__main__":
    main()
