import os
from pathlib import Path
import networkx as nx
import numpy as np
from scipy.stats import wasserstein_distance
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description='Select subnetworks based on distance.')
parser.add_argument('--network_id', type=str, required=True, help='Network ID')
parser.add_argument('--n_reps', type=int, default=10,
                    help='Number of repetitions')
args = parser.parse_args()

network_id = args.network_id
n_reps = args.n_reps

root = Path(f'test_data/networks/syn/{network_id}')

orig_subnetworks_dir = root / 'orig_subnetworks'
syn_subnetworks_dir = root / 'syn_subnetworks_raw'

out_root = root / 'syn_subnetworks'

syn_subnetworks = []
syn_cluster_id2iid = dict()
syn_cluster_iid2id = dict()

for subnetwork_fp in syn_subnetworks_dir.iterdir():
    if subnetwork_fp.suffix != '.tsv':
        continue

    subnetwork_id = subnetwork_fp.stem
    subnetwork_iid = len(syn_subnetworks)
    syn_cluster_id2iid[subnetwork_id] = subnetwork_iid
    syn_cluster_iid2id[subnetwork_iid] = subnetwork_id

    with open(subnetwork_fp, 'r') as f:
        edges = [tuple(line.strip().split('\t')) for line in f]
    syn_subnetworks.append(edges)

orig_subnetworks = []
orig_cluster_id2iid = dict()
orig_cluster_iid2id = dict()

for subnetwork_fp in orig_subnetworks_dir.iterdir():
    if subnetwork_fp.suffix != '.tsv':
        continue

    subnetwork_id = subnetwork_fp.stem
    subnetwork_iid = len(orig_subnetworks)
    orig_cluster_id2iid[subnetwork_id] = subnetwork_iid
    orig_cluster_iid2id[subnetwork_iid] = subnetwork_id

    with open(subnetwork_fp, 'r') as f:
        edges = [tuple(line.strip().split('\t')) for line in f]
    orig_subnetworks.append(edges)


def distance(G1, G2):
    deg1 = np.array([d for _, d in G1.degree()])
    deg2 = np.array([d for _, d in G2.degree()])
    return wasserstein_distance(deg1, deg2)


pbar = tqdm(total=len(orig_subnetworks))
d = np.zeros((len(orig_subnetworks), len(syn_subnetworks)))
for i, orig_subnetwork in enumerate(orig_subnetworks):
    pbar.set_description(f'Orig {i}')
    pbar.update(1)
    for j, syn_subnetwork in enumerate(syn_subnetworks):
        orig_G = nx.Graph(orig_subnetwork)
        syn_G = nx.Graph(syn_subnetwork)
        d[i, j] = distance(orig_G, syn_G)
p = np.exp(-d)
p = p / p.sum(axis=1, keepdims=True)

pbar = tqdm(range(n_reps))
for rep in pbar:
    pbar.set_description(f'Rep {rep}')

    chosen_subnetworks = dict()

    for i, orig_subnetwork in enumerate(orig_subnetworks):
        # Select randomly based on the distance (the smaller the better)
        selected_iid = np.random.choice(len(syn_subnetworks), p=p[i])
        selected_id = syn_cluster_iid2id[selected_iid]
        chosen_subnetworks[orig_cluster_iid2id[i]] = selected_id

    out_dir = out_root / f'rep_{rep}'
    out_dir.mkdir(parents=True, exist_ok=True)

    for orig_id, syn_id in chosen_subnetworks.items():
        syn_fp = syn_subnetworks_dir / f'{syn_id}.tsv'
        out_fp = out_dir / f'{orig_id}_{syn_id}.tsv'
        os.system(f'cp {syn_fp} {out_fp}')
