"""
Data loader for RNet

Authors: Lukas Moeller, Lorenzo Guerci
01/2022
"""



import torch, os
import pandas as pd
from torch.utils.data import Dataset
from rna_binding.utils import (
    pickle_load,
    MODE_DICT,
    BASE_PATH
)



class LOADER(Dataset):
    def __init__(self, path, mode=0):
        # open list with names of molecules that should be loaded
        mol_dict = pd.read_csv(path, delimiter=',', header=None)
        mol_names = mol_dict[0].tolist()

        # set variables
        self.mol_names = [name for name in mol_names]
        self.mode = mode
        
    def __len__(self):
        return len(self.mol_names)

    def __getitem__(self, idx):
        # get mol id
        name = self.mol_names[idx]

        # load pre-computed molecule grids
        mol, target = pickle_load(os.path.join(BASE_PATH, 'data/mol', str(MODE_DICT[self.mode]),'voxel', str(name) + '.pkl'))
        
        return torch.Tensor(mol), torch.Tensor(target)


class PREDICTION_LOADER(Dataset):
    def __init__(self, path, mode=0):
        # open list with names of molecules that should be loaded
        mol_dict = pd.read_csv(path, delimiter=',', header=None)
        mol_names = mol_dict[0].tolist()

        # set variables
        self.mol_names = [name for name in mol_names]
        self.mode = mode
        
    def __len__(self):
        return len(self.mol_names)

    def __getitem__(self, idx):
        # get mol id
        name = self.mol_names[idx]

        # load pre-computed molecule grids
        mol, target = pickle_load(os.path.join(BASE_PATH, 'data/mol', str(MODE_DICT[self.mode]),'voxel', str(name) + '.pkl'))
        
        return torch.Tensor(mol), torch.Tensor(target), name



def main():
    import time
    import argparse
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()
    parser.add_argument("-bs", type=int, required=False, default=4, help='batch size')
    parser.add_argument("-nw", type=int, required=False, default=1, help='number of workers')
    parser.add_argument("-path", type=str, required=False, default='data/crossval/prot/all/train.csv') 
    parser.add_argument("-mode", type=int, required=False, default=1, help='running mode: 0 = RNA, 1 = Protein')
    args = parser.parse_args()

    train_data = LOADER(os.path.join(BASE_PATH, args.path), args.mode)
    train_loader = DataLoader(train_data, batch_size=args.bs, num_workers=args.nw, shuffle=False)

    starttime = time.time()
    print("STARTED LOOP")
    for mol, target in train_loader:
        pass
    total_time = time.time() - starttime
    print(total_time)
    print("COMPLETED LOOP")


if __name__ == "__main__":
    main()
