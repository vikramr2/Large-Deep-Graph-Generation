#!/usr/bin/env python3
import networkx as nx
import argparse
import pandas as pd
import random

def main():
    parser = argparse.ArgumentParser(description='Run NetworkX label propagation clustering')
    parser.add_argument('input_file', help='Input TSV file containing edgelist')
    parser.add_argument('output_file', help='Output TSV file for cluster assignments')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
    
    # Read edgelist with integer node IDs
    edges_df = pd.read_csv(args.input_file, sep='\t', header=None, names=['source', 'target'], dtype=int)
    
    # Create NetworkX graph
    G = nx.from_pandas_edgelist(edges_df, 'source', 'target')
    
    # Run label propagation
    communities = nx.community.label_propagation_communities(G)
    
    # Convert generator to list of sets and enumerate communities
    community_dict = {}
    for i, community in enumerate(communities):
        for node in community:
            community_dict[node] = i
    
    # Create output dataframe
    output_df = pd.DataFrame.from_dict(community_dict, orient='index', columns=['cluster'])
    output_df.index.name = 'node'
    output_df.reset_index(inplace=True)
    
    # Sort by node ID
    output_df = output_df.sort_values('node')
    
    # Save results without headers
    output_df.to_csv(args.output_file, sep='\t', header=False, index=False)

if __name__ == "__main__":
    main()