import os
import shutil
import unicodedata

mapping_filename = 'twitch_gamers_kaffpa_cc.tsv'
edge_filename = 'twitch_gamers_cleaned.tsv'

dataset_name = os.path.commonprefix([mapping_filename, edge_filename]).rstrip('_')

def normalize_node_id(s):
    s = s.strip()
    s = unicodedata.normalize('NFC', s)
    s = ''.join(c for c in s if c.isprintable())
    return s

def sanitize_filename(s):
    forbidden_chars = r'<>:"/\\|?*'
    s = ''.join(c for c in s if c not in forbidden_chars)
    s = s.strip()
    if not s:
        s = 'unnamed_subgraph'
    return s

if os.path.exists('subgraphs'):
    shutil.rmtree('subgraphs')

os.makedirs('subgraphs')

node_to_subgraph = {}
with open(mapping_filename, 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f, 1):
        parts = line.strip().split()
        if len(parts) != 2:
            print(f"Skipping invalid line {line_num} in mapping file: {line.strip()}")
            continue  
        node_id_str, subgraph_id_str = parts
        node_id = normalize_node_id(node_id_str)
        subgraph_id = subgraph_id_str.strip()
        node_to_subgraph[node_id] = subgraph_id

subgraph_files = {}

with open(edge_filename, 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f, 1):
        parts = line.strip().split()
        if len(parts) != 2:
            print(f"Skipping invalid line {line_num} in edge file: {line.strip()}")
            continue  
        node1 = normalize_node_id(parts[0])
        node2 = normalize_node_id(parts[1])
        if node1 in node_to_subgraph and node2 in node_to_subgraph:
            subgraph_id1 = node_to_subgraph[node1]
            subgraph_id2 = node_to_subgraph[node2]
            if subgraph_id1 == subgraph_id2:
                subgraph_id = subgraph_id1
                sanitized_subgraph_id = sanitize_filename(subgraph_id)
                filename = f"{sanitized_subgraph_id}.tsv"
                filepath = os.path.join('subgraphs', filename)
                if subgraph_id not in subgraph_files:
                    try:
                        subgraph_files[subgraph_id] = open(filepath, 'w', encoding='utf-8')
                    except OSError as e:
                        print(f"Error opening file '{filepath}': {e}")
                        continue
                subgraph_files[subgraph_id].write(f"{node1}\t{node2}\n")

for f in subgraph_files.values():
    f.close()

print(f"Number of unique subgraph IDs: {len(subgraph_files)}")
print(f"Number of subgraph files created: {len(subgraph_files)}")

shutil.make_archive(dataset_name, 'zip', 'subgraphs')
print(f"Zip file '{dataset_name}.zip' created.")
