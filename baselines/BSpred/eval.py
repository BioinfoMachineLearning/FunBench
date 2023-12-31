import subprocess
import os
from Bio import SeqIO


import ray

# Initialize Ray
ray.init()

@ray.remote
def process_record(record, save_dir):
    try:
        pdb_id = record.id
        sequence = str(record.seq)
        pdb_dir = os.path.join(save_dir, pdb_id)
        os.makedirs(pdb_dir, exist_ok=True)
        # write the sequence to pdb_dir
        res_path = os.path.join(pdb_dir, 'prediction.txt')
        if os.path.exists(res_path):
            if os.path.getsize(res_path) != 0:
                return
        with open(os.path.join(pdb_dir, 'seq.txt'), 'w') as f:
            f.write(f">{pdb_id}" + "\n" + sequence)
        subprocess.run(['./INSTALL_BSpred/bin/runBSpred.pl', pdb_dir])

    except KeyboardInterrupt as e:
        raise e
    except:
        # write to file
        with open('error_pdb_id.txt', 'a') as f:
            f.write(pdb_id + '\n')

def run_infer_ls(records, save_dir):
    if not os.path.exists('error_pdb_id.txt'):
        with open('error_pdb_id.txt', 'w') as f:
            f.write('')

    futures = [process_record.remote(record, save_dir) for record in records]
    ray.get(futures)  # Wait for all tasks to complete


def run_infer_path(a3m, save_dir):
    # read a3m file using bio
    with open(a3m, 'r') as f:
        records = list(SeqIO.parse(f, 'fasta'))
    run_infer_ls(records, save_dir)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--a3m', type=str, default='../../construct_testset/data/next_pdb_protein_seq_non_redundant.fasta')
    parser.add_argument('--save_dir', type=str, default='./predictions')
    args = parser.parse_args()
    run_infer_path(args.a3m, args.save_dir)

