import subprocess, argparse, os
import pandas as pd
from rna_binding.utils import BASE_PATH, MODE_DICT



if __name__ == "__main__":

    # argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", type=int, required=False, default=0, help='running mode: 0 = RNA, 1 = Protein')
    parser.add_argument("-num", type=int, required=False, default=20, help='subprocess number')
    args = parser.parse_args()

    # output path
    save_path = os.path.join(BASE_PATH, 'data/mol', str(MODE_DICT[args.mode]), 'voxel/multiprocessing')
    os.makedirs(save_path, exist_ok=True)

    # read molecules to voxelize
    if args.mode == 0:
        dataset_df = pd.read_csv(os.path.join(BASE_PATH, 'data/dataset_preparation/rna/rna_dataset_filter2.csv'), delimiter=',', index_col=None) 
        mol_ids = dataset_df['pdb id'].values
    else:
        dataset_df = pd.read_csv(os.path.join(BASE_PATH, 'data/dataset_preparation/prot/prot_dataset_filter.csv'), delimiter=',', index_col=None) 
        mol_ids = dataset_df['scpdb id'].values

    split_num = round(len(mol_ids)/args.num)
    for i in range(args.num):
        if i < args.num - 1:
            tmp_mols = mol_ids[split_num*i:split_num*i+split_num]
        else:
            tmp_mols = mol_ids[split_num*i:]
        with open(os.path.join(save_path,'subprocess'+str(i)+'.txt'), 'w') as f_out:
            for mol in tmp_mols:
                f_out.write(mol + '\n')

    # submission of multiple processes
    for i in range(args.num):
        cmd_str = [
            "bsub",
            "-n",
            "1",
            "-W",
            "4:00",
            "-sp",
            "90",
            "-R",
            "rusage[mem=10240]",
            "python",
            "descriptor_generation/descriptor_calculation.py",
            "-mode",
            str(args.mode),
            "-path",
            BASE_PATH + '/' + 'data/mol/' + str(MODE_DICT[args.mode]) + '/voxel/multiprocessing/subprocess' + str(i) + '.txt'
        ]

        completed_process = subprocess.run(cmd_str)