import os
import pickle
import networkx as nx

folder_path = "/content/drive/MyDrive/amazon"

graphs = []

for file_name in sorted(os.listdir(folder_path)):
    if file_name.endswith(".tsv"):
        file_path = os.path.join(folder_path, file_name)

        # Create a new graph for each file
        graph = nx.Graph()

        with open(file_path, "r") as file:
            for line in file:
                node1, node2 = map(int, line.strip().split("\t"))  # Parse nodes from the line
                graph.add_edge(node1, node2, label='DEFAULT_LABEL')  # Add edge with 'label'

        # Add the 'label': 'DEFAULT_LABEL' attribute to all nodes
        nx.set_node_attributes(graph, 'DEFAULT_LABEL', name='label')

        # Append the graph to the list
        graphs.append(graph)

output_path = "/content/drive/MyDrive/amazon.pkl"
with open(output_path, "wb") as pkl_file:
    pickle.dump(graphs, pkl_file)

print(f"Serialized {len(graphs)} graphs to {output_path}")
