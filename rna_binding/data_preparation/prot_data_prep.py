"""
Protein Dataset Preparation

Author: Lukas Moeller
Date: 01/2022
"""



import os, glob, logging, warnings
import numpy as np
import pandas as pd
from moleculekit.molecule import Molecule
from moleculekit.util import boundingBox
from moleculekit.smallmol.smallmol import SmallMol
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem import MolFromSmiles
from rna_binding.utils import (
    check_frames,
    pickle_save,
    BASE_PATH,
    ELEMENT_MASS,
    ELEMENTS_CH,
    ELEMENTS_LIG,
    BOX_SIZE,
    LOWER_LIG_MASS,
    UPPER_LIG_MASS
)



def store_lig_mass(lig, lig_info_df, kwd, val):
    '''
    store type of ligand in df
    
    Args:
        ligands (list): list of ligands to be added
        lig_info_df (pd.DataFrame): see description below

    Returns:
        lig_info_df (pd.DataFrame): see description below
    '''
    # check if ligand already available in df
    if lig in lig_info_df['lig id'].values:
        lig_info_df.loc[lig_info_df['lig id'] == lig, kwd] = val
    # add ligand to dataframe
    else:
        lig_info_row = pd.DataFrame(zip([lig], [None], [None]), columns=['lig id', 'mass', 'mass status'])
        lig_info_row[kwd] = val
        lig_info_df = lig_info_df.append(lig_info_row)
    return lig_info_df


def calc_ligand_mass(small_mol, mol, ligands, lig_info_df):
    '''
    calculate the molecular mass (in Da) of a ligands

    Args:
        mol (moleculekit.Molecule): molecule of interest
        ligands (list): list of ligands for which mass should be calculated
        lig_info_df (pd.DataFrame): see description below
    
    Returns:
        lig_info_df (pd.DataFrame): see description below
    '''
    # check if mass was already calculated previously
    new_ligands = np.setdiff1d(ligands, lig_info_df.loc[lig_info_df['mass'].notna(), 'lig id'].values)

    # single ligand present
    if len(new_ligands) == 1 and len(ligands) == 1 and small_mol is not None:
        # get smiles string of ligand
        smiles = small_mol.toSMILES()

        # get molecule object from smiles string
        mol_from_smiles = MolFromSmiles(smiles)
        
        # get exact ligand mass
        exact_mass = ExactMolWt(mol_from_smiles)

        # store that exact ligand mass calculation was possible
        lig_info_df = store_lig_mass(ligands[0], lig_info_df, 'mass status', 'exact')

        # store calculated ligand mass
        lig_info_df = store_lig_mass(ligands[0], lig_info_df, 'mass', exact_mass)

    # multiple ligands present
    else:
        # loop over ligands with unknown mass
        for lig in new_ligands:

            # parameter determines if exact mass should be calculated or mass should be approximated: set to exact mass
            do_approximation = False

            # get smiles string of ligand
            smiles = smiles_df[smiles_df[1] == lig]

            # if smiles string available, calculate exact mass
            if not smiles.empty:
                smiles = smiles.iat[0, 0]

                # check if smiles string has valid type to prevent errors
                if type(smiles) == str:

                    # get molecule object from smiles string
                    mol_from_smiles = MolFromSmiles(smiles)
                    
                    # check if molecule object could be generated from smiles string to prevent errors
                    if mol_from_smiles is not None:
                        # get exact ligand mass
                        exact_mass = ExactMolWt(mol_from_smiles)
                        # store that exact ligand mass calculation was possible
                        lig_info_df = store_lig_mass(lig, lig_info_df, 'mass status', 'exact')
                    else:
                        do_approximation = True
                
                # if smiles string is of invalid type, do approximation of ligand mass
                else:
                    do_approximation = True
            
            # if smiles string not available, do approximation of mass
            else:
                do_approximation = True
            
            # approximate mass of ligand if parameter set to true
            if do_approximation is True:
                # do approximation
                try:
                    exact_mass = approximate_ligand_mass(mol, lig)
                # set mass to zero if approximation fails (for large molecules if moleculekit filtering function raises runtime error)
                except RuntimeError:
                    exact_mass = 0
                
                # if approximation successful, store info that ligand mass was approximated
                if exact_mass > 0:
                    lig_info_df = store_lig_mass(lig, lig_info_df, 'mass status', 'approximation')
                # if approximation failed, store this information
                else:
                    lig_info_df = store_lig_mass(lig, lig_info_df, 'mass status', 'error')
        
            # store calculated ligand mass
            lig_info_df = store_lig_mass(lig, lig_info_df, 'mass', exact_mass)
    
    return lig_info_df


def approximate_ligand_mass(mol, lig):
    '''
    approximate the molecular mass (in Da) of a ligands

    Args:
        mol (moleculekit.Molecule): molecule of interest
        lig (str): ligands id of ligand for which mass should be calculated
    
    Returns:
        lig_mass (float): approximated ligand mass in Da
    '''
    # get ligands with corresponding name from molecule
    lig_filter = mol.copy()
    lig_filter.filter('resname ' + str(lig))
    # select first ligand with corresponding name
    first_res_id = lig_filter.resid[0]
    lig_filter.filter('resid ' + str(first_res_id))

    # loop over atoms of ligand and add atom masss
    ligand_mass = 0
    for atom in lig_filter.element:
        if atom in ELEMENTS_LIG:
            ligand_mass += ELEMENT_MASS[atom]
    
    # since hydrogen atoms are not stored in moleculekit objects, their influence on the total ligand mass has to be approximated
    return ligand_mass * 1.05


def warn(*args, **kwargs):
    """
    silence warnings
    """
    pass



if __name__ == '__main__':

    # silence warnings / info
    logging.getLogger('moleculekit.molecule').setLevel(logging.ERROR)
    logging.getLogger('moleculekit.smallmol.smallmol').setLevel(logging.ERROR)
    logging.getLogger('__name__').setLevel(logging.ERROR)
    warnings.warn = warn

    # load paths to pdb files
    scpdb_path = os.path.join(BASE_PATH, 'data/mol/prot/scpdb/*')
    scpdb_files = glob.glob(scpdb_path)

    # load smiles strings for all pdb ligands
    # file "Components-smiles-stereo-oe.smi" can be downloaded from pdb: http://ligand-expo.rcsb.org/ld-download.html
    smiles_path = os.path.join(BASE_PATH, 'data/dataset_preparation/Components-smiles-stereo-oe.smi')
    smiles_df = pd.read_csv(smiles_path, delimiter='\t', header=None, on_bad_lines='warn')

    # define & create directory to save results
    out_path = os.path.join(BASE_PATH, 'data/dataset_preparation/prot')
    os.makedirs(out_path, exist_ok=True)

    # initialize empty dataframes to store results from data processing
    #
    # note: lists will also be stored in dataframes to generate a user-friendly overview over the dataset in a table-like format
    #
    # dataset_df: stores if pdb entry ('scpdb id') fullfills filtering criterium (1) or not (0)
    # filtering criteria:
    #  - Ligand mass of enclosed ligands ('lig mass'): specifies if at least one ligand with suitable mass (200 - 700 Da) present
    #  - Elements present in scpdb entry ('elements'): specifies if only elements present in protein structure (not ligands) that are 
    #    part of the 8-channel encoding used to generate descriptors (only relevant for usecase examined in this project not for usage
    #    of dataset in general)
    #  - Grid site of scpdb entry ('grid size'): specifies if structure fits in cube with edgelength of 80 Angström (only relevant for
    #    usecase examined in this project not for usage of dataset in general)
    dataset_df = pd.DataFrame(columns=['scpdb id', 'lig mass', 'elements', 'grid size'])
    
    # dataset_info_df: stores information relevant for filtering of pdb entries ('scpdb id')
    # information:
    #  - List of all ligand ids (unsuitable and suitable for training): 'lig' (duplicate/... ids only listed once)
    #  - List of all suitable ligands (suitable mass): 'lig filtered'
    #  - Number of suitable ligands (suitable mass): 'lig num'
    #  - Number of structure type labels == 1 for molecule: 'type num'
    #  - List of elements present in molecule: 'elements'
    #  - List of elements not encoded by RNet but present in molecule: 'elements out'
    #  - Edge length of cube used to describe molecule to train RNet in dimension 1: 'size dim1'
    #  - Edge length of cube used to describe molecule to train RNet in dimension 2: 'size dim2'
    #  - Edge length of cube used to describe molecule to train RNet in dimension 3: 'size dim3'
    #  - Minimal edge length of cube describing molecule: maximal value of 3 cube dimensions described before ('size')
    dataset_info_df = pd.DataFrame(columns=['scpdb id', 'lig', 'lig filtered', 'lig num', 'elements', 'elements out', 'size dim1',
                                        'size dim2', 'size dim3', 'size'])
    
    # lig_info_df: stores information relevant for assessing suitablility of ligands
    # information:
    #  - 'lig id': residue name of ligand (ids as used in pdb)
    #  - 'mass': molecular mass of ligand in Da
    #  - 'mass status': specifies if molecular mass exact value (exact) or approximation (approximation) or if error during mass
    #    calculation occured (error: if error: mass = 0)
    lig_info_df = pd.DataFrame(columns=['lig id', 'mass', 'mass status'])
    
    # initialize dictionaries
    lig_dict = {} # cf. column 'lig' in dataset_info_df
    lig_filted_dict = {} # cf. column 'lig filtered' in dataset_info_df

    # number of molecules to be processed
    mol_num = len(scpdb_files)

    # loop over molecules for filtering
    for i, file_path in enumerate(scpdb_files):

        # ################################################## 0. load ligand & protein ##################################################
        # load scpdb entry id
        mol_name = os.path.basename(os.path.realpath(os.path.splitext(file_path)[0]))

        # logging
        print('loaded molecule', str(mol_name), '|', str(round((i+1)/mol_num*100)), 'percent completed')

        # load ligand
        try:
            lig = SmallMol(os.path.join(file_path, 'ligand.mol2'), force_reading=True)
        except (ValueError, Exception) as e:
            lig = None
        lig_mol = Molecule(os.path.join(file_path, 'ligand.mol2'))        

        # load protein
        mol_mol = Molecule(os.path.join(file_path, 'protein.mol2'))

        # keep single frame
        check_frames(lig_mol)
        check_frames(mol_mol)
        
        # initialize new rows for dataframes
        dataset_row = pd.DataFrame(zip([''], [1], [1], [1]), columns=['scpdb id', 'lig mass', 'elements', 'grid size'])
        dataset_info_row = pd.DataFrame(columns=['scpdb id', 'lig', 'lig filtered', 'lig num', 'elements', 'elements out', 'size dim1',
                                                'size dim2', 'size dim3', 'size'])
        dataset_row['scpdb id'] = [mol_name]
        dataset_info_row['scpdb id'] = [mol_name]

        # ################################# 1. filter structures according to mol. mass of ligands ###################################
        # calculate mass of all ligands
        ligands = list(np.unique(lig_mol.resname))
        lig_info_df = calc_ligand_mass(lig, lig_mol, ligands, lig_info_df)
        
        # get ligands with suitable mass for pbd entry
        suitable_lig_mass = [key for key in ligands if LOWER_LIG_MASS <= lig_info_df.loc[lig_info_df['lig id'] == key, 'mass'].iloc[0]
                <= UPPER_LIG_MASS]
        
        # store results
        dataset_info_row['lig'] = [ligands]
        lig_dict[mol_name] = ligands
        dataset_info_row['lig filtered'] = [suitable_lig_mass]
        lig_filted_dict[mol_name] = suitable_lig_mass
        dataset_info_row['lig num'] = len(suitable_lig_mass)
        
        # check if at least one ligand with suitable mass, otherwise sort out pdb entry
        if len(suitable_lig_mass) == 0:
            dataset_row['lig mass'] = [0]

        # ####################################### 2. filter structures according to element occurrance #################################
        elements = np.unique(mol_mol.element)
        dataset_info_row['elements'] = [elements]
        elements = np.setdiff1d(elements, ELEMENTS_CH + ['H'])
        dataset_info_row['elements out'] = [elements]
        if len(elements) > 0:
            dataset_row['elements'] = [0]


        # ######################################## 3. filter structures according to grid size ##########################################
        # get bounding box of molecule
        bbox = boundingBox(mol_mol)

        # calculate edge length of bounding box
        max_diameter = np.linalg.norm(bbox, axis=0)

        # store information
        dataset_info_row['size dim1'] = max_diameter[0]
        dataset_info_row['size dim2'] = max_diameter[1]
        dataset_info_row['size dim3'] = max_diameter[2]
        dataset_info_row['size'] = np.max(max_diameter)
        
        # sort out molecule if does not fit in BOX_SIZE
        if np.max(max_diameter) > BOX_SIZE:
            dataset_row['grid size'] = [0]

        # ################################################### append to data frame #####################################################
        # append rows to data frames
        dataset_df = dataset_df.append(dataset_row)
        dataset_info_df = dataset_info_df.append(dataset_info_row)
        
        # store temporary results
        dataset_df.to_csv(os.path.join(out_path, 'prot_dataset_tmp.csv'), sep=',', header=True, index=False)
        dataset_info_df.to_csv(os.path.join(out_path, 'prot_dataset_info_tmp.csv'), sep=',', header=True, index=False)
        lig_info_df.to_csv(os.path.join(out_path, 'lig_dataset_info_tmp.csv'), sep=',', header=True, index=False)


    # save all results

    # save dictionaries
    result_dict = {
        'lig': lig_dict,
        'lig_filtered': lig_filted_dict
    }
    # save to pkl file
    pickle_save(os.path.join(out_path, 'prot_result_dicts.pkl'), result_dict)

    # save dataframes
    dataset_df.to_csv(os.path.join(out_path, 'prot_dataset.csv'), sep=',', header=True, index=False)
    dataset_info_df.to_csv(os.path.join(out_path, 'prot_dataset_info.csv'), sep=',', header=True, index=False)
    lig_info_df.to_csv(os.path.join(out_path, 'lig_dataset_info.csv'), sep=',', header=True, index=False)
    
    # filter for suitable scpdb entries
    dataset_df = dataset_df.astype({'lig mass': int, 'elements': int, 'grid size': int})
    dataset_df['filter sum'] = dataset_df.iloc[:,1:4].sum(axis=1)
    dataset_df_f1 = dataset_df[dataset_df['filter sum'] >= 3]
    dataset_df_f1.to_csv(os.path.join(out_path, 'prot_dataset_filter.csv'), sep=',', header=True, index=False)
