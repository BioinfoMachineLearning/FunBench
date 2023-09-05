import pandas as pd
from glob import glob
import os
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import subprocess


def get_pdb(pdb_code="", save_dir="pdbs"):
    if os.path.isfile(pdb_code):
        return pdb_code
    elif len(pdb_code) == 4:
        os.system(
            f"wget -qnc https://files.rcsb.org/view/{pdb_code}.pdb -O {save_dir}/{pdb_code}.pdb"
        )
        return f"{pdb_code}.pdb"
    else:
        os.system(
            f"wget -qnc https://alphafold.ebi.ac.uk/files/AF-{pdb_code}-F1-model_v3.pdb"
        )
        return f"AF-{pdb_code}-F1-model_v3.pdb"


def run_infer_ls(pdb_id_list, out_dir):
    if not os.path.exists("error_pdb_id.txt"):
        with open("error_pdb_id.txt", "w") as f:
            f.write("")

    for pdb_id in pdb_id_list:
        try:
            code, chain = pdb_id.split("_")
            res_dir = os.path.join(out_dir, pdb_id)
            os.makedirs(res_dir, exist_ok=True)
            get_pdb(code, res_dir)
            if len(glob(os.path.join(res_dir, "*.csv"))) != 0:
                print(f"{pdb_id} already predicted.")
                continue
            subprocess.run(
                [
                    "python",
                    "prediction.py",
                    "--querypath",
                    res_dir,
                    "--filename",
                    code + ".pdb",
                    "--chainid",
                    chain,
                    "--ligands",
                    "DNA,RNA,CA,MG,MN,ATP,HEME",
                    "--cpu",
                    "10",
                ]
            )
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            # write to file
            with open("error_pdb_id.txt", "a") as f:
                f.write(pdb_id + "\n")


def run_infer_path(label_h5_path, out_dir):
    pdb_id_list = load_ids(label_h5_path)
    run_infer_ls(pdb_id_list, out_dir)

def load_ids(label_h5_path):
    """
    return a dataframe with columns ['id','true_binding_sites'], 
    where true_binding_sites is a list of binding sites for each type molecules.
    
    """
    true_df = pd.read_hdf(label_h5_path, key='df')
    # return id column as a list
    return true_df['id'].values.tolist()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--label_h5_path_list',
                        default=['/home/zpw97/github-projs/PDBAnnotator/test_df_CA_binding.h5'],
                        nargs='+',
                    help='the h5 file containing the true labels.')
    parser.add_argument("--out_dir", type=str, default="../eval_results")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for label_h5_path in args.label_h5_path_list:
        run_infer_path(label_h5_path, args.out_dir)
