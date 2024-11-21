import pandas as pd
from sys import argv

data = pd.read_csv(argv[1], compression='gzip')

# split the file into two parts: name and extension
name, ext = argv[1].rsplit('.', 1)
name, ext = name.rsplit('.', 1)

# save the decompressed file
data.to_csv(name + '.tsv', index=False, sep='\t')
