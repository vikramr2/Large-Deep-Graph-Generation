import zipfile
import pandas as pd
from collections import defaultdict

graph_file = "twitch_gamers_cleaned.tsv"
subgraph_file = "twitch_gamers_kaffpa_cc.tsv"
output_zip = "twitch-gamers_subgraphs.zip"

graph_df = pd.read_csv(graph_file, sep="\t", header=None, names=["node_i", "node_j"])
subgraph_df = pd.read_csv(subgraph_file, sep="\t", header=None, names=["node", "subgraph"])
node_to_subgraph = subgraph_df.set_index("node")["subgraph"].to_dict()
subgraph_data = defaultdict(list)

for _, row in graph_df.iterrows():
    node_i, node_j = row["node_i"], row["node_j"]
    subgraph_i = node_to_subgraph.get(node_i)
    subgraph_j = node_to_subgraph.get(node_j)
    
    if subgraph_i:
        subgraph_data[subgraph_i].append(f"{node_i}\t{node_j}\n")
    if subgraph_j and subgraph_j != subgraph_i:
        subgraph_data[subgraph_j].append(f"{node_i}\t{node_j}\n")

with zipfile.ZipFile(output_zip, "w") as zf:
    for subgraph, edges in subgraph_data.items():
        file_name = f"{subgraph}.tsv"
        zf.writestr(file_name, "".join(edges))

print(f"Subgraphs written to {output_zip}")
