# Evaluation of GraphBind

## Installation

Running the install script to install conda env, required packages, and download binary tools blast and hhsuite.
```
bash install.sh
```
### Download Database

GraphBind requires a database Uniref50 for blast, please download it by running the following command:

```
cd scripts
wget https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz
gzip -d uniref50.fasta.gz # 解压
makeblastdb -in uniref50.fasta -parse_seqids -hash_index -dbtype prot # compile database
```

And then download Uniref30 for sequence alignment by hhsuite:
```
wget https://gwdu111.gwdg.de/~compbiol/uniclust/2023_02/UniRef30_2023_02_hhsuite.tar.gz
tar xzf UniRef30_2023_02_hhsuite.tar.gz

```

## Setup 

Set the absolute paths of HHblits and uniclust databases in the script "scripts/prediction.py".

## Inference

Running eval.py to infer the binding sites of the test set. Specify the test pdb ids by `--pdb_id_path` and the saved path by 
`--out_dir`. 

```
conda activate GraphBind_env
cd scripts

python eval.py --label_h5_path_list /home/zpw97/github-projs/PDBAnnotator/test_df_rna_binding.h5
python eval.py --label_h5_path_list /home/zpw97/github-projs/PDBAnnotator/test_df_dna_binding.h5
python eval.py --label_h5_path_list /home/zpw97/github-projs/PDBAnnotator/test_df_ATP_binding.h5
python eval.py --label_h5_path_list /home/zpw97/github-projs/PDBAnnotator/test_df_CA_binding.h5
python eval.py --label_h5_path_list /home/zpw97/github-projs/PDBAnnotator/test_df_MG_binding.h5
python eval.py --label_h5_path_list /home/zpw97/github-projs/PDBAnnotator/test_df_MN_binding.h5
```

## Evaluation

```
conda activate pdb_anno
py compute_metrics.py --binding_task ATP --label_h5_path /home/zpw97/github-projs/PDBAnnotator/test_df_ATP_binding.h5
py compute_metrics.py --binding_task DNA --label_h5_path /home/zpw97/github-projs/PDBAnnotator/test_df_dna_binding.h5
py compute_metrics.py --binding_task RNA --label_h5_path /home/zpw97/github-projs/PDBAnnotator/test_df_rna_binding.h5
py compute_metrics.py --binding_task Ca2+ --label_h5_path /home/zpw97/github-projs/PDBAnnotator/test_df_CA_binding.h5
py compute_metrics.py --binding_task Mg2+ --label_h5_path /home/zpw97/github-projs/PDBAnnotator/test_df_MG_binding.h5
py compute_metrics.py --binding_task Mn2+ --label_h5_path /home/zpw97/github-projs/PDBAnnotator/test_df_MN_binding.h5
```
