import argparse
from io import StringIO
import os
from Bio import SeqIO
import subprocess
import pandas as pd
import requests
# from build_dataset import build_dataset
from src.utils import read_tab_separated_file

def save_mol_protein_sequences(pdb_seq_path, output_path,savedir):
    if not os.path.exists(pdb_seq_path):
        opath = download_file('https://ftp.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt.gz', savedir)
        subprocess.run(["gunzip", opath])
        
    if os.path.exists(output_path):
        with open(output_path) as pdb_seq_file:
            protein_records = list(SeqIO.parse(pdb_seq_file, "fasta"))
            return protein_records
    
    # Read the PDB sequence data using Biopython's SeqIO
    with open(pdb_seq_path) as pdb_seq_file:
        records = list(SeqIO.parse(pdb_seq_file, "fasta"))

        # Filter the records to only include mol:protein sequences
        protein_records = [record for record in records if "mol:protein" in record.description]

    # Write the filtered protein sequences to the output FASTA file
    with open(output_path, "w") as output_file:
        SeqIO.write(protein_records, output_file, "fasta")
    return protein_records

def download_file(url, save_dir):
        """Download file from url to save_dir. return the filepath."""
        file_name = os.path.join(save_dir, os.path.basename(url))
        if os.path.exists(file_name):
            return file_name
        
        os.makedirs(save_dir, exist_ok=True)
        response = requests.get(url)
        response.raise_for_status()
    

        with open(file_name, "wb") as file:
            file.write(response.content)

        return file_name
    
def parse_entries(
    savedir,
    deposition_date_url='https://files.wwpdb.org/pub/pdb/derived_data/index/entries.idx'
    ):
        """
        Parse the PDB entries file and return a dictionary mapping PDB IDs to
        deposition dates.
        """
        # download from url
        deposition_date_fpath = download_file(deposition_date_url, savedir)
        with open(deposition_date_fpath, "r") as f:
            lines = f.readlines()
        lines = lines[2:]  # Skip header
        # Note: There's a badly formatted line we need to deal with instead of
        # using Pandas' builtin CSV parser.
        lines = [l.replace('"', "") for l in lines]

        df = pd.read_csv(
            StringIO("".join(lines)),
            sep="\t",
            header=None,
            skipinitialspace=True,
        )
        df.columns = [
            "id",
            "name",
            "date",
            "title",
            "source",
            "authors",
            "resolution",
            "experiment_type",
        ]
        df.dropna(subset=["id"], inplace=True)

        df.id = df.id.str.lower()
        df.date = pd.to_datetime(df.date)
        return df


def split_and_save_data(dataframe, threshold_date, prev_save_path, next_save_path):
    # Assuming the columns are "date" and "pdbid"
    if "date" not in dataframe.columns or "id" not in dataframe.columns:
        print("Error: 'date' or 'id' column not found in the DataFrame.")
        raise KeyError("Error: 'date' or 'pdbid' column not found in the DataFrame.")
    
    # Convert the date column to datetime
    dataframe["date"] = pd.to_datetime(dataframe["date"])
    
    # Split into prev and next dataframes
    prev_dataframe = dataframe[dataframe["date"] <= threshold_date]
    next_dataframe = dataframe[dataframe["date"] > threshold_date]
    
    # Save pdbid columns to text files
    prev_pdbids = prev_dataframe["id"].tolist()
    next_pdbids = next_dataframe["id"].tolist()
    
    with open(prev_save_path, 'w') as prev_file:
        prev_file.write('\n'.join(prev_pdbids))
    
    with open(next_save_path, 'w') as next_file:
        next_file.write('\n'.join(next_pdbids))

    return prev_pdbids, next_pdbids

def fasta2idtxt(input_fasta, output_txt):
    sequence_ids = []

    # Parse the FASTA file and extract sequence IDs
    with open(input_fasta, 'r') as fasta_file:
        for record in SeqIO.parse(fasta_file, 'fasta'):
            sequence_ids.append(record.id)

    # Save the sequence IDs to the output text file
    with open(output_txt, 'w') as txt_file:
        txt_file.write('\n'.join(sequence_ids))

    print(f"Sequence IDs saved to {output_txt}")

def seq_search(
    query_path,
    target_path,
    search_result_path,
    tmp_dir = './data/tmp'
    ):
    
    db_dir = './data/db'
    os.makedirs(tmp_dir, exist_ok=True); os.makedirs(db_dir, exist_ok=True)
    
    # Step 1: Create MMseqs2 databases for query and target
    query_db,target_db, result_db = f"{db_dir}/queryDB", f"{db_dir}/targetDB", f"{db_dir}/resultDB"
    subprocess.run(["mmseqs", "createdb", query_path, query_db,])
    subprocess.run(["mmseqs", "createdb", target_path, target_db,])

    # Step 2: Run sequence search
    subprocess.run(["mmseqs", "search", query_db, target_db, result_db, tmp_dir])

    # Step 3: Convert result database to BLAST tab format
    subprocess.run(["mmseqs", "convertalis", query_db, target_db, result_db, search_result_path])

    # Step 4: Extract non-redundant sequences
    search_df = read_tab_separated_file(search_result_path)
    search_df.to_csv(search_result_path, index=False)
 
    
    return search_df


        

def save_sequences_by_ids(fasta_file_path, sequence_ids, output_file_path):
    sequence_id_set = set(sequence_ids)
    
    with open(fasta_file_path, "r") as fasta_file, open(output_file_path, "w") as output_file:
        batch_size = 1000  # You can adjust this based on your system's memory
        batch = []
        
        for record in (r for r in SeqIO.parse(fasta_file, "fasta") if r.id.split("_")[0] in sequence_id_set):
            batch.append(record)
            if len(batch) >= batch_size:
                SeqIO.write(batch, output_file, "fasta")
                batch = []
        if batch:
            SeqIO.write(batch, output_file, "fasta")


def parse_args():
    parser = argparse.ArgumentParser(description="Process PDB data")
    parser.add_argument("--savedir", default="data", help="Directory to save data files")
    parser.add_argument("--date_threshold", default="2022-01-01", help="Date threshold")
    parser.add_argument("--threshold_identity", type=float, default=0.3, help="Threshold identity")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    savedir = args.savedir
    date_threshold = args.date_threshold
    threshold_identity = args.threshold_identity

    
    pdb_seqres_path = f'{savedir}/pdb_seqres.txt'
    protein_seq_path = f'{savedir}/pdb_protein_seq.fasta'
    
    prev_seq_path = f'{savedir}/prev_pdbids_{date_threshold}.fasta'
    next_seq_path = f'{savedir}/next_pdbids_{date_threshold}.fasta'
    pre_pdbid_path = f'{savedir}/prev_pdbids_{date_threshold}.txt'
    next_pdbid_path = f'{savedir}/next_pdbids_{date_threshold}.txt'
    search_result_path = f'{savedir}/next2prev-search_result_seq-id-{threshold_identity}.csv'
    
    output_seq_path = f'{savedir}/nonredundant_thre-{date_threshold}_seq-id-{threshold_identity}.fasta'
    output_id_path = f'{savedir}/nonredundant_thre-{date_threshold}_seq-id-{threshold_identity}.txt'
    
    
    protein_records = save_mol_protein_sequences(pdb_seq_path=pdb_seqres_path, output_path=protein_seq_path,savedir=savedir)
    release_df = parse_entries(savedir)
    prev_pdbids, next_pdbids = split_and_save_data(release_df, date_threshold,
                                                  pre_pdbid_path,
                                                  next_pdbid_path)

    save_sequences_by_ids(protein_seq_path, prev_pdbids, prev_seq_path)
    save_sequences_by_ids(protein_seq_path, next_pdbids, next_seq_path)

    search_df = seq_search(next_seq_path,
                                           prev_seq_path, search_result_path)
    
    redundant_pdbs = search_df[search_df["sequence_identity"] >= threshold_identity]["query_sequence"].tolist()
    non_redundant_pdbs = list(set(next_pdbids) - set(redundant_pdbs))
    # save non_redundant_pdbs to file
    with open(output_id_path, 'w') as f:
        f.write('\n'.join(non_redundant_pdbs))
    
    save_sequences_by_ids(protein_seq_path, non_redundant_pdbs, output_seq_path)