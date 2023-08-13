import pandas as pd



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

    