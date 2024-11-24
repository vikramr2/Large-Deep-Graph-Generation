import matplotlib.pyplot as plt

# Read the file
with open('tmppartition70', 'r') as file:
    data = file.readlines()

# Convert data to integers
data = [int(line.strip()) for line in data]

# Create a histogram
plt.hist(data, bins=range(71), edgecolor='black', align='left')

# Add titles and labels
plt.title('Cluster Sizes in tmppartition70')
plt.xlabel('Number')
plt.ylabel('Frequency')

# Show the plot
plt.savefig('kaffpa70_cit-hepPh_histogram.png')
