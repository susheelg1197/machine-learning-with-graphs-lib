import networkx as nx
from ml_wg.classification.wl_kernel import weisfeiler_lehman_kernel

def main():
    # Create two example graphs
    G1 = nx.gnp_random_graph(10, 0.5, seed=42)
    G2 = nx.gnp_random_graph(10, 0.5, seed=43)

    # Compute the Weisfeiler-Lehman kernel between G1 and G2
    kernel_value = weisfeiler_lehman_kernel(G1, G2, h=3)
    print("Weisfeiler-Lehman Kernel between G1 and G2:", kernel_value)

if __name__ == "__main__":
    main()
