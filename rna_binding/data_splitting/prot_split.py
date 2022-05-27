import os, random
import pandas as pd
from rna_binding.utils import BASE_PATH



if __name__ == "__main__":
    # load molecule names
    dataset_df = pd.read_csv(os.path.join(BASE_PATH, 'data/dataset_preparation/prot/prot_dataset_filter.csv'), delimiter=',', index_col=None) 
    mol_ids = dataset_df['scpdb id'].values

    # load output path
    save_path = os.path.join(BASE_PATH + '/data/crossval/prot', 'all')
    os.makedirs(save_path, exist_ok=True)

    # shuffle list
    random.shuffle(mol_ids)

    # split list in two sets
    split_num = int(round(len(mol_ids)*0.9))

    # save all mols
    with open(os.path.join(save_path, 'all.csv'), 'w') as f_out:
        for i in range(len(mol_ids)):
            f_out.write(str(mol_ids[i]) + '\n')

    # save train set
    with open(os.path.join(save_path, 'train.csv'), 'w') as f_out:
        for i in range(split_num):
            f_out.write(str(mol_ids[i]) + '\n')

    # save eval set
    with open(os.path.join(save_path, 'eval.csv'), 'w') as f_out:
        for i in range(split_num, len(mol_ids)):
            f_out.write(str(mol_ids[i]) + '\n')
