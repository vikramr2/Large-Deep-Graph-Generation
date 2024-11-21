import pandas as pd
from sys import argv


file, ext = argv[1].split('/')[-1].split('.')

df = pd.read_csv(argv[1])
df.to_csv(f'{file}.tsv', sep='\t', index=False)
