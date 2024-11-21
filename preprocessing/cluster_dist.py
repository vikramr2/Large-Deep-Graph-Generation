import pandas as pd
import matplotlib.pyplot as plt
from sys import argv

# Load the data
edgelist = pd.read_csv(argv[1], sep="\t", header=None)

clusters = pd.read_csv(argv[2], sep="\t", header=None)

print(len(clusters[1].unique()))

# The first column is nodes and the second column is cluster assignments, get the number of nodes in each cluster
cluster_sizes = clusters[1].value_counts()

# Remove 10% smallest clusters and 10% largest clusters
# n_clusters = len(cluster_sizes)
# n_remove = int(n_clusters * 0.1)
# cluster_sizes = cluster_sizes.sort_values()
# cluster_sizes = cluster_sizes.iloc[n_remove:-n_remove]

# Remove clusters with only one node
cluster_sizes = cluster_sizes[cluster_sizes > 2]

# Plot the distribution of cluster sizes
plt.hist(cluster_sizes, bins=100)
plt.yscale('log')
plt.xlabel("Cluster size")
plt.ylabel("Frequency")
plt.title("Distribution of cluster sizes")
plt.savefig(f"{argv[2]}.png")

# Get a cumulative distribution of cluster sizes as a line plot
# cluster_sizes = cluster_sizes.sort_values()
# cumulative_sizes = cluster_sizes.cumsum()
# cumulative_sizes = cumulative_sizes / cumulative_sizes.max()
# plt.plot(cumulative_sizes)
# plt.xlabel("Cluster index")
# plt.ylabel("Cumulative fraction of nodes")
# plt.title("Cumulative distribution of cluster sizes")
# plt.savefig("cumulative_cluster_sizes.png")
