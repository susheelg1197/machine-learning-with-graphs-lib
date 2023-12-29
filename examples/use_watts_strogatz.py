from ml_wg.graph_generation.watts_strogatz import generate_watts_strogatz
import networkx as nx
import matplotlib.pyplot as plt

def visualize_graph(G):
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray')
    plt.title("Watts-Strogatz Graph")
    plt.show()

def main():
    n, k, p = 20, 4, 0.3
    G = generate_watts_strogatz(n, k, p, add_weights=True, add_attributes=True)
    visualize_graph(G)

if __name__ == "__main__":
    main()
