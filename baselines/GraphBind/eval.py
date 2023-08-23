import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os

def get_pdb(pdb_code=""):
    if os.path.isfile(pdb_code):
        return pdb_code
    elif len(pdb_code) == 4:
        os.system(f"wget -qnc https://files.rcsb.org/view/{pdb_code}.pdb")
        return f"{pdb_code}.pdb"
    else:
        os.system(f"wget -qnc https://alphafold.ebi.ac.uk/files/AF-{pdb_code}-F1-model_v3.pdb")
        return f"AF-{pdb_code}-F1-model_v3.pdb"
    