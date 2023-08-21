# Evaluatin of CLAPE


# Installation
CLAPE is build up on `pytorch`, 
you can using an existing pytorch environment and additionally install following packages by: 
```
pip install biopython transformers tqdm
```

# Running Predictions

Activate the environment: `conda activate pestocus`

Running clape to infer DNA and RNA binding sites on the test sequences 
`../../construct_testset/data/next_pdb_protein_seq_non_redundant.fasta`,
 and output the results to `DNA_sites.txt` and `RNA_sites.txt`:

```
python clape.py --input ../../construct_testset/data/next_pdb_protein_seq_non_redundant.fasta\
 --output DNA_sites.txt --ligand DNA --device cuda:0
python clape.py --input ../../construct_testset/data/next_pdb_protein_seq_non_redundant.fasta\
 --output RNA_sites.txt --ligand RNA --device cuda:1
```

