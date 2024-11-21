EL=$1
cleaned=`basename $EL .tsv`_cleaned.tsv
clustering=`basename $EL .tsv`_0.001_clustering.tsv

Rscript cleanup_el.R $EL $cleaned
python3 run_leiden.py -i $cleaned -r 0.001 -o $clustering -n 2
python3 cluster_dist.py $cleaned $clustering
