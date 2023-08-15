import pandas as pd


import numpy as np
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd

def read_pesto_release_date():
    def plot_deposition_date_distribution(dataframe1, dataframe2):
        # Assuming the column name is "deposition_date" in both dataframes
        if "deposition_date" not in dataframe1.columns or "deposition_date" not in dataframe2.columns:
            print("Error: 'deposition_date' column not found in one or both of the DataFrames.")
            return
        
        # Convert the deposition_date column to datetime
        dataframe1["deposition_date"] = pd.to_datetime(dataframe1["deposition_date"])
        dataframe2["deposition_date"] = pd.to_datetime(dataframe2["deposition_date"])
        
        # Plot the distribution
        plt.figure(figsize=(10, 6))
        plt.hist(dataframe1["deposition_date"], bins=30, color='skyblue', edgecolor='black', alpha=0.5, label='pesto data')
        plt.hist(dataframe2["deposition_date"], bins=30, color='salmon', edgecolor='black', alpha=0.5, label='non-pesto data')
        plt.title("Distribution of Deposition Date")
        plt.xlabel("Deposition Date")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    ids = set()
    for path in glob("/home/zpw97/projs/PeSTo/datasets/subunits_*.txt"):
        with open(path) as f:
            for line in f.readlines():
                ids.add(line.strip().split('_')[0].lower())
        print(f"done reading {path}")

    release_info = pd.read_csv('construct_testset/data/release_date.csv')

    pesto_release_info = release_info[release_info['pdbid'].isin(ids)]
    non_pesto_release_info = release_info[~release_info['pdbid'].isin(ids)]

    plot_deposition_date_distribution(pesto_release_info,non_pesto_release_info)
    print("training data cutoff date:", pesto_release_info['deposition_date'].max())





def read_tab_separated_file(file_path):
    # Define column names
    columns = [
        "query_sequence",
        "target_sequence",
        "sequence_identity",
        "alignment_length",
        "num_mismatches",
        "num_gap_openings",
        "query_domain_start",
        "query_domain_end",
        "target_domain_start",
        "target_domain_end",
        "e_value",
        "bit_score"
    ]

    # Read the file into a DataFrame
    df = pd.read_csv(file_path, sep='\t', header=None, names=columns)
    return df

    