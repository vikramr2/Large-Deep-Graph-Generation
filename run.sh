network_id="amazon0302"
n_reps=100

# python graph_partition.py \
#     --network_id $network_id

# python clean_outlier.py \
#     --input-network test_data/networks/orig/${network_id}/${network_id}_cleaned.tsv \
#     --input-clustering test_data/networks/orig/${network_id}/${network_id}_kaffpa_cc.tsv \
#     --output-folder test_data/networks/wo-outliers/${network_id}/

# python subnetwork_selection.py \
#     --network_id ${network_id} \
#     --n_reps ${n_reps}

python subnetwork_stitching.py \
    --network_id ${network_id}

python compute_stats.py \
    --input-network test_data/networks/wo-outliers/${network_id}/${network_id}_cleaned.tsv \
    --input-clustering test_data/networks/wo-outliers/${network_id}/${network_id}_kaffpa_cc.tsv \
    --output-folder test_data/stats/wo-outliers/${network_id}/

for i in $(seq 0 $((n_reps-1)));
do
    python compute_stats.py \
        --input-network test_data/networks/syn/${network_id}/syn_networks/rep_${i}/network.tsv \
        --input-clustering test_data/networks/syn/${network_id}/syn_networks/rep_${i}/clustering.tsv \
        --output-folder test_data/stats/syn/${network_id}/rep_${i}/ \
        --overwrite

    python compare_stats_pair.py \
        --network-1-folder test_data/stats/wo-outliers/${network_id}/ \
        --network-2-folder test_data/stats/syn/${network_id}/rep_${i}/ \
        --output-file test_data/stats/syn/${network_id}/rep_${i}/compare.csv
done

python plot_distances.py \
    --root_dir test_data/stats/syn/${network_id} \
    --output_dir test_data/plots/${network_id}