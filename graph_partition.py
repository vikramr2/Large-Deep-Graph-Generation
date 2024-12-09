import csv
from pathlib import Path
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='Partition a graph into subnetworks.')
    parser.add_argument('--network_id', type=str, required=True,
                        help='ID of the network to partition')
    return parser.parse_args()


args = parse_args()
network_id = args.network_id

inp_network_fp = f'test_data/networks/orig/{network_id}/{network_id}_cleaned.tsv'
inp_clustering_fp = f'test_data/networks/orig/{network_id}/{network_id}_kaffpa_cc.tsv'

out_dir = Path(f'test_data/networks/syn/{network_id}/orig_subnetworks')
out_dir.mkdir(parents=True, exist_ok=True)

# Load clustering
all_clusters = set()
clustered_nodes = set()
node2cluster = dict()
cluster2nodes = dict()
with open(inp_clustering_fp, 'r') as f:
    reader = csv.reader(f, delimiter='\t')
    for v, c in reader:
        assert v not in clustered_nodes
        node2cluster[v] = c
        cluster2nodes.setdefault(c, set()).add(v)
        clustered_nodes.add(v)
        all_clusters.add(c)

# Load edgelist
all_nodes = set()
all_cc_edges = set()
all_co_edges = set()
all_oo_edges = set()
with open(inp_network_fp, 'r') as f:
    reader = csv.reader(f, delimiter='\t')
    for u, v in reader:
        all_nodes.add(u)
        all_nodes.add(v)
        if u in clustered_nodes and v in clustered_nodes:
            all_cc_edges.add((u, v))
        elif u in clustered_nodes or v in clustered_nodes:
            all_co_edges.add((u, v))
        else:
            all_oo_edges.add((u, v))

# Load outliers
outlier_nodes = all_nodes - clustered_nodes

subnetworks = dict()
n_edges_per_cluster = dict()
for c in all_clusters:
    subnetworks[c] = set()
    # Ignore all edges with at least one outlier
    for u, v in all_cc_edges:
        if u in outlier_nodes or v in outlier_nodes:
            continue
        if u in cluster2nodes[c] and v in cluster2nodes[c]:
            subnetworks[c].add((u, v))
    n_edges_per_cluster[c] = len(subnetworks[c])

# Export subnetworks
for c, edges in subnetworks.items():
    out_fp = out_dir / f'{c}.tsv'
    with open(out_fp, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        for u, v in edges:
            writer.writerow([u, v])
    print(f'Exported {len(edges)} edges to {out_fp}')
