import torch
from torch_geometric.data import Data
from ml_wg.gnn.gcn_md import GCNMD  # Assuming the GCNMD class is in the gcn_md.py file

def generate_sample_data(num_nodes, in_channels_modality1, in_channels_modality2):
    # Generating random graph data for the purpose of this example
    edge_index = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=torch.long).t().contiguous()
    x_modality1 = torch.randn((num_nodes, in_channels_modality1))  # Demographic features
    x_modality2 = torch.randn((num_nodes, in_channels_modality2))  # Behavioral features
    return Data(x_modality1=x_modality1, x_modality2=x_modality2, edge_index=edge_index)

def main():
    num_nodes = 10
    in_channels_modality1 = 5  # Number of demographic features
    in_channels_modality2 = 3  # Number of behavioral features
    out_channels = 2  # Number of output classes (e.g., will buy, will not buy)

    # Generate sample data
    data = generate_sample_data(num_nodes, in_channels_modality1, in_channels_modality2)

    # Initialize the GCN-MD model
    model = GCNMD(in_channels_modality1, in_channels_modality2, out_channels)

    # Apply the model to the data
    output = model(data)

    # Output interpretation
    print("Predictions for each node:")
    for i in range(num_nodes):
        print(f"Node {i}: {output[i]}")

if __name__ == "__main__":
    main()
