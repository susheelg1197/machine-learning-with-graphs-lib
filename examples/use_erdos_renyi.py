from ml_wg.graph_generation.erdos_renyi import generate_erdos_renyi
import networkx as nx
import matplotlib.pyplot as plt

def main():
    n = 50
    p = 0.2
    G = generate_erdos_renyi(n, p, connected=True, weights=True, seed=42)
    
    # Visualize the generated graph
    pos = nx.spring_layout(G)
    weights = nx.get_edge_attributes(G, 'weight')
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edgelist=weights.keys(), edge_color=list(weights.values()), edge_cmap=plt.cm.Blues)
    plt.show()

    # Use the graph for further analysis or machine learning tasks
    # ...

if __name__ == "__main__":
    main()
