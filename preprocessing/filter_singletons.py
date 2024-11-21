''' 
Inputs:
- Edgelist file: input network edge-list path
- Clustering assignment file: output membership path

Finds nodes that don't belong to any cluster and removes them 
from the network as well as the clustering assignment file.
'''

import pandas as pd
from sys import argv


edgelist = pd.read_csv(argv[1], sep="\t", header=None)
clustering = pd.read_csv(argv[2], sep="\t", header=None)