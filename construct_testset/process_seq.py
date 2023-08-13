import os
from Bio import SeqIO
import subprocess
import pandas as pd


def save_mol_protein_sequences(pdb_seq_path='data/pdb_seqres.txt', output_path='data/pdb_protein_seq.fasta'):
    if not os.path.exists(pdb_seq_path):
        raise FileNotFoundError("Unable to find the PDB sequence file.")
    
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


def remove_redundant_sequences(
    query_path,
    target_path,
    search_result_path,
    filter_result_path,
    seq_identity=0.3,
    tmp_dir = './data/tmp'
    ):
    
    db_dir = './data/db'
    os.makedirs(tmp_dir, exist_ok=True); os.makedirs(db_dir, exist_ok=True)
    
    try:
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
        filter_df = search_df[search_df['sequence_identity'] < seq_identity]
        filter_df.to_csv(filter_result_path, index=False)
    except Exception as e:
        print("Error:", str(e))




if __name__ == '__main__':
    protein_records = save_mol_protein_sequences(pdb_seq_path='data/pdb_seqres.txt', output_path='data/pdb_protein_seq.fasta')

