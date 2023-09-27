import subprocess
import os

def run_infer_ls(pdb_id_list):
    if not os.path.exists('error_pdb_id.txt'):
        with open('error_pdb_id.txt', 'w') as f:
            f.write('')
            
    for pdb_id in pdb_id_list:
        p = f'output/all_feat_3l/pred_data/pred_{pdb_id}.npy'
        if os.path.exists(p): pass
        try:
            subprocess.run(['./data_prepare_one.sh', pdb_id, ])
            subprocess.run(['./predict_site.sh', pdb_id, ])
        except KeyboardInterrupt as e:
            raise e
        except:
            # write to file
            with open('error_pdb_id.txt', 'a') as f:
                f.write(pdb_id + '\n')
                
                
def run_infer_path(pdb_id_path,):
    with open(pdb_id_path, 'r') as f:
        pdb_id_list = f.readlines()
        pdb_id_list = list(set([pdb_id.strip() for pdb_id in pdb_id_list]))
    run_infer_ls(pdb_id_list)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--pdb_id_path', type=str, default='/home/zpw97/github-projs/PDBAnnotator/test_pdb_protein_binding.txt')

    args = parser.parse_args()
    run_infer_path(args.pdb_id_path)
