import pandas as pd
import subprocess

def df2fasta(df, fastapath):
    """df of columns id and seq, save it as fasta file"""
    with open(fastapath,'w') as f:
        for i in range(len(df)):
            f.write(f'>{df["id"][i]}\n')
            f.write(f'{df["seq"][i]}\n')
    return None

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--label_h5_path',default='/home/zpw97/github-projs/PDBAnnotator/test_df_dna_binding.h5',
                    help='the h5 file containing the true labels.')
parser.add_argument('--ligand', '-l', type=str, help='Ligand type, including DNA, RNA, and antibody',
                   default='DNA', choices=['DNA', 'RNA', 'AB'])
parser.add_argument('--threshold', '-t', type=float, help='Threshold of classification score', default=0.5)
parser.add_argument('--output', '-o', help='Output file path, default clape_result.txt',
                   default='DNA_sites.txt')
parser.add_argument('--cache', '-c', help='Path for saving cached pre-trained model', default='protbert')
parser.add_argument('--device', '-d', help='Device to run the model on', default='cuda')

args = parser.parse_args()

df2fasta(pd.read_hdf(args.label_h5_path, key='df'), 'infer.fasta')
# python clape.py --input ../../construct_testset/data/next_pdb_protein_seq_non_redundant.fasta --output DNA_sites.txt --ligand DNA --device cuda:0
subprocess.run(['python', 'clape.py', '--input', 'infer.fasta', '--output', args.output, '--ligand', args.ligand, '--device', args.device])