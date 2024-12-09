import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

STATS_OF_INTEREST = [
    ("deg_assort", "scalar", "abs_diff"),
    ("global_ccoeff", "scalar", "abs_diff"),

    ("n_nodes", "scalar", "rel_diff"),
    ("n_edges", "scalar", "rel_diff"),
    ("diameter", "scalar", "rel_diff"),

    ("degree", "distribution", "emd"),
    ("local_ccoeff", "distribution", "emd"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze distance distributions from multiple runs.")
    parser.add_argument("--root_dir", type=str,
                        help="Path to the root directory containing the rep_i directories.")
    parser.add_argument("--output_dir", type=str, default="figures",
                        help="Directory to save the generated figures.")
    return parser.parse_args()


def analyze_distances(root_dir, output_dir):
    all_data = []
    for rep_dir in os.listdir(root_dir):
        if rep_dir.startswith("rep_"):
            csv_path = os.path.join(root_dir, rep_dir, "compare.csv")
            if os.path.exists(csv_path):
                try:  # Handle potential errors in individual CSV files
                    rep_data = pd.read_csv(csv_path)
                    all_data.append(rep_data)
                except pd.errors.ParserError as e:
                    print(f"Error reading CSV file {csv_path}: {e}")
                    continue  # Skip this file and move to the next
            else:
                print(f"Warning: compare.csv not found in {rep_dir}")

    if not all_data:
        print("No data found. Exiting.")
        return

    combined_data = pd.concat(all_data, ignore_index=True)

    os.makedirs(output_dir, exist_ok=True)

    for stat in combined_data['stat'].unique():
        for stat_type in combined_data['stat_type'].unique():
            for distance_type in combined_data['distance_type'].unique():
                if (stat, stat_type, distance_type) not in STATS_OF_INTEREST:
                    continue

                subset = combined_data[
                    (combined_data['stat'] == stat) &
                    (combined_data['stat_type'] == stat_type) &
                    (combined_data['distance_type'] == distance_type)
                ]

                if not subset.empty:
                    plt.figure()
                    # Use seaborn for better visualization
                    sns.histplot(subset['distance'],
                                 kde=True, stat='probability')
                    # if max and min are opposite signs, add a vertical line at 0
                    if subset['distance'].max() * subset['distance'].min() < 0:
                        plt.axvline(0.0, color='red', linestyle='dotted',
                                    linewidth=2)
                    # plt.axvline(0.0, color='red', linestyle='dotted',
                    #             linewidth=2)  # Add vertical line
                    plt.title(
                        f"Distance Distribution for {stat} ({stat_type}, {distance_type})")
                    plt.xlabel("Distance")
                    plt.ylabel("Frequency")
                    plt.savefig(os.path.join(
                        output_dir, f"{stat}_{stat_type}_{distance_type}.pdf"))
                    plt.close()  # Close the figure to free memory


if __name__ == "__main__":
    args = parse_args()
    analyze_distances(args.root_dir, args.output_dir)
