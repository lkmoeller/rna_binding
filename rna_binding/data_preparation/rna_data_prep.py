"""
RNA Dataset Preparation

Author: Lukas Moeller
Date: 01/2022
"""



import os, glob, logging, re, warnings
import numpy as np
import pandas as pd
from moleculekit.molecule import Molecule
from moleculekit.util import boundingBox
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem import MolFromSmiles
from scipy.spatial import distance
from rna_binding.utils import (
    check_frames,
    reshape_coords,
    get_single_lig,
    decompose_mol,
    pickle_save,
    BASE_PATH,
    RNA_CODES,
    PROT_CODES,
    ELEMENTS_CH,
    ELEMENTS_LIG,
    ELEMENT_MASS,
    BOX_SIZE,
    LOWER_LIG_MASS,
    UPPER_LIG_MASS
)



# Threshold for fraction of RNA atoms per molecule (relative fraction of RNA to protein atoms) considered for the removal of RNA-
# protein structures from dataset
RNA_FRAC_THRESHOLD = 0.5
# Threshold for fraction of RNA atoms within binding site (relative fraction of RNA to protein atoms): considered for the removal of
# binding sites containing both RNA and protein residues
BINDING_SITE_FRAC_THRESHOLD = 0.5
# distance threshold (distance RNA/protein atom to nearest ligand atom) for calculation of binding site fraction of RNA atoms
BINDING_SITE_FRAC_DIST_THRESHOLD = 10
# list with ligands classified as detergents, buffer salts, or crystallization components by Rekand and Brenk, J. Chem. Inf. Model.
# 2021, 61, 8, 4068–4081
LIG_LIST = pd.read_csv(os.path.join(BASE_PATH, 'data/dataset_preparation/lig_codes_solvent_buffer.txt'))
# keywords for evaluation of RNA structure type: presence of keywords in pdb files evaluated for classification
# Structure types:
#  - 0: ribosomal RNA
#  - 1: viral RNA
#  - 2: ribozyme
#  - 3: riboswitch
#  - 4: aptamer (part of riboswitch, without expression platform)
#  - 5: primer-template complexes (not relevant for learning on RNA structures due to DNA-like configuration)
#  - 6: splicing-related structures
#  - 7: other structures (no keywords of list present in pdb file)
# multiple classifications possible
kwd_dict = {'RIBOSOM': 0, 'RIBOSOME': 0, 'RIBOSOMAL': 0, 'A SITE': 0, 'P SITE': 0, 'E SITE': 0, 'DECODING': 0,
            'AMINOGLYCOSIDE': 0, 'RRNA': 0, 'VIRAL': 1, 'VIRUS': 1, 'HIV': 1, 'MOSAIC': 1, 'CORONA': 1, 'RIBOZYME': 2,
            'RIBOSWITCH': 3, 'APTAMER': 4, 'MANGO': 4, 'CORN': 4, 'PRIMER': 5, 'MEIMPG': 5, 'METHYLIMIDAZOLE': 5,
            'POLYMERASE': 5, 'TRANSCRIPTION': 5, 'SIGNAL RECOGNITION PARTICLE': 5, 'SPLICING': 6, 'SPLICEOSOME': 6}
kwd_list = list(kwd_dict.keys())
rev_kwd_dict = {0: 'ribosomal rna', 1: 'viral rna', 2: 'ribozyme', 3: 'riboswitch', 4: 'aptamer', 5: 'primer complex',
            6: 'splicing-related', 7: 'other structure'}
# keywords for riboswitch classification
kwd_riboswitch = ['YKKC', 'YADO', 'ATP', 'NAD']



def calc_rna_frac(mol):
    '''
    calculate fraction of RNA atoms for molecule

    Args:
        mol (moleculekit.Molecule): molecule of interest
    
    returns:
        fraction of RNA atoms: float between 0 and 1
    '''
    rna = np.sum(mol.atomselect('resname ' + ' '.join(i for i in RNA_CODES))) # select all RNA atoms
    prot = np.sum(mol.atomselect('resname ' + ' '.join(i for i in PROT_CODES))) # select all protein atoms
    return rna/(rna + prot)


def get_ligands(mol):
    '''
    get all ligands (incl. metal ions, covalently bound ligands, non-standard amino acids/nucleic acids/biomolecules)

    Args:
        mol (moleculekit.Molecule): molecule of interest
    
    Returns:
        list of residue names for all ligands
    '''
    ligands = np.setdiff1d(np.unique(mol.resname), RNA_CODES) # remove all natural nucleic acids from list of molecule residue names
    ligands = np.setdiff1d(ligands, PROT_CODES) # remove all natural amino acids from list of molecule residue names
    return list(ligands)


def text_mining_pdb(path_2_pdb, ligand_list):
    '''
    text mining of pdb file to extract information about RNA structure type and types of enclosed ligands

    Args:
        path_2_pdb (os.path): path to pdb file of interest
        ligand_list (list): list of residue names that could potentially be ligands

    Returns:
        type_list (list): list of identifiers for RNA structure type
        keep_ligand_list (list): list of ligands after sorting out metal ions and covalently bound components
        metal_list (list): list of metal ions present in structure
        covalent_link_list (list): list of molecules covalentely bound to RNA/protein residues
        p_dict (dict): dictionary storing the number of P-atoms for each ligand - key: ligand id, value: number of P-atoms (max. 4)
        ribo_list (list): list of keywords allowing classification of riboswitches
    '''

    # initialize empty lists / dictionaries
    type_list = [] # list of structure type identifier for pdb entry
    metal_list = [] # list of metal ions
    covalent_link_list = [] # list of molecules covalentely bound to RNA/protein residues
    p_dict = {} # number of P-atoms for each ligand for given pdb entry - key: ligand id, value: number of P-atoms (max. 4)
    ribo_list = [] # list of keywords that potentially enable riboswitch structure identification
    
    # open and mine pdb file
    with open(path_2_pdb, 'r') as f_in:
        for line in f_in:
            # keep only letters, numbers, and spaces
            line_list = re.sub('[^A-Z0-9 ]', ' ', line.upper()).split()

            # get identifier of line in pdb file (e. g. TITLE, ATOM, HETATM, etc.)
            line_id = line_list[0]

            # if ATOM section reached, all desired information processed: stop mining
            if line_id == 'ATOM':
                break
            
            # search for keywords allowing classification of RNA structure type in the sections TITLE, COMPND, KEYWDS, JRNL
            elif line_id in ['TITLE', 'COMPND', 'KEYWDS', 'JRNL']:
                line_list_kwd = np.intersect1d(line_list, kwd_list)
                line_types = [kwd_dict[x] for x in line_list_kwd]
                type_list = np.unique(np.concatenate((type_list, line_types), 0))
                line_list_ribo = np.intersect1d(line_list, kwd_riboswitch)
                ribo_list = np.unique(np.concatenate((ribo_list, line_list_ribo), 0))
            
            # search for all (metal) ions
            elif line_id == 'HETNAM':
                if line_list[-1] == 'ION':
                    metal_list.append(line_list[1])
            
            # extract and store number of P-atoms for all ligands (except metal ions)
            elif line_id == 'FORMUL':
                ligand_list = np.setdiff1d(ligand_list, metal_list)
                ligand_id = np.intersect1d(line_list, ligand_list)
                if len(ligand_id) > 0:
                    ligand_id = str(ligand_id[0])
                    if 'P' in line_list:
                        p_dict[ligand_id] = 1
                    elif 'P2' in line_list:
                        p_dict[ligand_id] = 2
                    elif 'P3' in line_list:
                        p_dict[ligand_id] = 3
                    elif 'P*' in line_list:
                        p_dict[ligand_id] = 4
                    else:
                        p_dict[ligand_id] = 0

            # check if potential ligands are covalently bound to RNA/protein residues and remove if true
            elif line_id == 'LINK':
                ligand_list = np.setdiff1d(ligand_list, metal_list)
                remove_single_lig = np.intersect1d(line_list, ligand_list)
                if len(np.intersect1d(line_list, metal_list)) == 0 and len(remove_single_lig) > 0:
                    covalent_link_list += list(remove_single_lig)
        
        # remove covalently bound molecules from ligand list
        if len(covalent_link_list) > 0:
            keep_ligand_list = np.setdiff1d(ligand_list, covalent_link_list)
        else:
            keep_ligand_list = ligand_list

    return list(type_list), list(keep_ligand_list), list(metal_list), list(covalent_link_list), p_dict, list(ribo_list)


def filter_structure_type(type_list, dataset_row, dataset_info_row):
    '''
    adjust and filter RNA structure types

    Args:
        type_list (list): list of RNA structure types identifiers
        dataset_row (pd.DataFrame): see description below
        dataset_info_row (pd.DataFrame): see description below

    Returns:
        type_list (list): list of RNA structure types identifiers
        dataset_row (pd.DataFrame): see description below
        dataset_info_row (pd.DataFrame): see description below
    '''
    # add 'other structure' label if no type
    if len(type_list) == 0:
        type_list.append(7)

    # correct riboswitch labels: remove aptamer label from riboswitches since all riboswitches contain aptamer & expression platform
    if 3 in type_list and 4 in type_list:
        type_list.remove(4)

    # correct virus labels: classify only as viral RNA if no other label available
    if 1 in type_list and len(type_list) > 1:
        type_list.remove(1)
    
    # filtering of RNA structure types: remove primer-template complexes
    if 5 in type_list:
        dataset_row['structure type'] = 0
        dataset_row['lig type'] = 0
    
    # store types to info df
    for i in type_list:
        dataset_info_row[rev_kwd_dict[i]] = 1
    for i in np.setdiff1d([0,1,2,3,4,5,6,7], type_list):
        dataset_info_row[rev_kwd_dict[i]] = 0
    dataset_info_row['type num'] = len(type_list)
    
    return type_list, dataset_row, dataset_info_row


def store_lig_type(ligands, lig_info_df, reason):
    '''
    store type of ligand in df
    
    Args:
        ligands (list): list of ligands to be added
        lig_info_df (pd.DataFrame): see description below
        reason (str): reason for (not) considering ligands as suitable, must match column titles of lig_info_df
        
    Returns:
        lig_info_df (pd.DataFrame): see description below
    '''
    for lig in ligands:
        # check if ligand already available in df
        if lig in lig_info_df['lig id'].values:
            lig_info_df.loc[lig_info_df['lig id'] == lig, reason] += 1
        # add ligand to dataframe
        else:
            lig_info_row = pd.DataFrame(zip([lig], [None], [None], [0], [0], [0], [0], [0], [0], [0], [0], [0]), columns=['lig id', 'mass',
                        'mass status', 'suitable lig', 'suitable type', 'NMP/NDP/NTP', 'water', 'complex', 'small mol', 'solvent/buffer', 'linker', 'covalent'])
            lig_info_row[reason] += 1
            lig_info_df = lig_info_df.append(lig_info_row)
    return lig_info_df


def rm_ligands(ligand_list, remove_ids, lig_info_df, reasoning):
    '''
    removes ligands with remove_ids from ligand_list and stores ligand type of removed ligands

    Args:
        ligand_list (list): list of ligands that should be checked for unsuitable ligands
        remove_ids (list): list of ligand ids that should be removed
        lig_info_df (pd.DataFrame): see description below
        reasoning (str): reasoning for removing ligands, must match column titles of lig_info_df 

    Returns:
        ligand_list (list): list of potentially suitable ligands
        lig_info_df (pd.DataFrame): see description below
    '''
    # get unsuitable ligands from ligand_list
    remove_ligands = np.intersect1d(ligand_list, remove_ids)
    # remove unsuitable ligands and store type
    if len(remove_ligands) > 0:
        ligand_list = np.setdiff1d(ligand_list, remove_ligands)
        lig_info_df = store_lig_type(remove_ligands, lig_info_df, reasoning)
    return list(ligand_list), lig_info_df


def filter_ligand_type(dataset_row, dataset_info_row, lig_info_df, ligands, link_list, p_dict, ribo_list):
    '''
    automatic and manual filtering of ligand types

    Args:
        dataset_row (pd.DataFrame): see description below
        dataset_info_row (pd.DataFrame): see description below
        lig_info_df (pd.DataFrame): see description below
        ligands (list): list of ligands that should be filtered according to their type
        link_list (list): list of molecules covalentely bound to RNA/protein residues
        p_dict (dict): dictionary storing the number of P-atoms for each ligand - key: ligand id, value: number of P-atoms (max. 4)
        ribo_list (list): list of keywords allowing classification of riboswitches

    returns:
        dataset_row (pd.DataFrame): see description below
        dataset_info_row (pd.DataFrame): see description below
        lig_info_df (pd.DataFrame): see description below
        ligand_list (list): filtered list of ligands
    '''

    # explanation filtering strategy:
    # 
    # automatic filtering of ligands: remove NMP-, NDP-, NTP-like ligands
    #   1.  remove always: 'G', 'GMP', 'GDP', 'GTP', 'C', 'CMP', 'CTP', 'U', 'UMP', 'UDP', 'UTP', 'T', 'TMP',
    #       'TDP', 'TTP', 'A', 'AMP'
    #       Note: 'GDP', 'GTP' could possibly control riboswitches due to their similarity to the alarmone ppGpp
    #       (M.E. Sherlock et al., Proceedings of the National Academy of Sciences Jun 2018, 115 (23) 6052-6057),
    #       but are excluded here to prevent noise in the data
    #   2.  if any riboswitch/aptamer: allow 'GUN' (Guanine), 'GNG' (2-Deoxyguanosine), 'ADE' (Adenine)
    #   3.  if ykkC riboswitch: allow ADP, dADP ('DAT'), CDP, and dCDP ('YYY') (M.E. Sherlock et al., Biochemistry,
    #       2019, 58, 5, 401-410)
    #   4.  if ydaO riboswitch/ATP-sensing riboswitch: allow ATP (P. Watson et. al., Nat Chem Biol, 2012, 8, 963–965)
    #   5.  if NAD+ riboswitch: allow ADP, ATP (H. Chen et al., Nucleic Acids Research, 2020, 48, 21, 12394–12406)
    #   6.  if any riboswitch/aptamer: allow ligands containing P, otherwise not
    #
    # manual filtering of ligands
    #   7.  water: HOH
    #   8.  crystallization artifacts/additive/buffers/solvents: 'EPE', 'SPM', 'DMS', 'IPA', 'MPD', 'NHE', 'B3P',
    #       'MPO', 'TCE', 'SPK' + ligands listed in Rekand and Brenk, J. Chem. Inf. Model. 2021, 61, 8, 4068–4081
    #   9.  metal-bound complexes: 'NCO', 'B1Z'
    #   10. small molecules with low mass/metabolites/amino acid derivatives: 'SIN', 'PHD', '9YL', 'MYR', 'DST',
    #       'BTB', 'NEG', 'N'
    #   11. linker-like structures: '2PE', 'S9L', 'MRC', '1N7', 'PE4', '1PE', 'PG6'

    # automatic removal steps
    # step (1)
    # removal criteria
    remove_ids_1 = ['G', 'GMP', 'GDP', 'GTP', 'C', 'CMP', 'CTP', 'U', 'UMP', 'UDP', 'UTP', 'T', 'TMP', 'TTP', 'A', 'AMP', 'DA',
                    'DC', 'DCP', 'DCT', 'DG', 'DGI', 'DGT', 'DT', 'TYD', 'DU', 'DUD', 'DUT', 'DAD']
    # remove ligands and store reason for removal
    ligand_list, lig_info_df = rm_ligands(ligands, remove_ids_1, lig_info_df, 'NMP/NDP/NTP')
    
    # step (2 - 6)
    # remove all ligands from steps 2 - 6 if structure is not a riboswitch or aptamer
    if not (3 in type_list or 4 in type_list):
        # removal criteria step 2 - 5
        remove_ids_2 = ['GUN', 'GNG', 'ADE', 'DAT', 'CDP', 'YYY', 'ADP', 'ATP']
        # remove ligands and store reason for removal
        ligand_list, lig_info_df = rm_ligands(ligand_list, remove_ids_2, lig_info_df, 'NMP/NDP/NTP') # step (2 - 5)
        # step 6: remove ligands containing P-atoms
        for k, v in p_dict.items():
            # check removal criteria
            if v > 1 and k not in remove_ids_1 and k not in remove_ids_2 and k not in link_list:
                ligand_list.remove(k) # step (6)
                # remove ligands and store reason for removal
                lig_info_df = store_lig_type(ligand_list, lig_info_df, 'NMP/NDP/NTP')
    
    # ligand removal scheme for riboswitches/aptamers
    else:
        if 'YKKC' not in ribo_list: # step (3)
            ligand_list, lig_info_df = rm_ligands(ligand_list, ['DAT', 'CDP', 'YYY'], lig_info_df, 'NMP/NDP/NTP')
        if 'NAD' not in ribo_list and 'YKKC' not in ribo_list: # step (3, 5)
            ligand_list, lig_info_df = rm_ligands(ligand_list, ['ADP'], lig_info_df, 'NMP/NDP/NTP')
        if 'YDAP' not in ribo_list and 'ATP' not in ribo_list and 'NAD' not in ribo_list: # step (4, 5)
            ligand_list, lig_info_df = rm_ligands(ligand_list, ['ATP'], lig_info_df, 'NMP/NDP/NTP')
    
    # manual removal steps
    # step (7)
    # removal criteria
    remove_man_1 = ['HOH']
    # remove ligands and store reason for removal
    ligand_list, lig_info_df = rm_ligands(ligand_list, remove_man_1, lig_info_df, 'water')
    
    # step (8)
    # removal criteria
    remove_man_2 = ['EPE', 'SPM', 'DMS', 'IPA', 'MPD', 'NHE', 'B3P', 'MPO', 'TCE', 'SPK']
    # remove ligands and store reason for removal
    ligand_list, lig_info_df = rm_ligands(ligand_list, remove_man_2, lig_info_df, 'solvent/buffer')
    # remove ligands classified as detergents, buffer salts, or crystallization components by
    # Rekand and Brenk, J. Chem. Inf. Model. 2021, 61, 8, 4068–4081
    remove_man_2 = LIG_LIST
    # remove ligands and store reason for removal
    ligand_list, lig_info_df = rm_ligands(ligand_list, remove_man_2, lig_info_df, 'solvent/buffer')
    
    # step (9)
    # removal criteria
    remove_man_3 = ['NCO', 'B1Z']
    # remove ligands and store reason for removal
    ligand_list, lig_info_df = rm_ligands(ligand_list, remove_man_3, lig_info_df, 'complex')
    
    # step (10)
    # removal criteria
    remove_man_4 = ['SIN', 'PHD', '9YL', 'MYR', 'DST', 'BTB', 'NEG', 'N']
    # remove ligands and store reason for removal
    ligand_list, lig_info_df = rm_ligands(ligand_list, remove_man_4, lig_info_df, 'small mol')
    
    # step (11)
    # removal criteria
    remove_man_5 = ['2PE', 'S9L', 'MRC', '1N7', 'PE4', '1PE', 'PG6']
    # remove ligands and store reason for removal
    ligand_list, lig_info_df = rm_ligands(ligand_list, remove_man_5, lig_info_df, 'linker')
    
    # check if ligand list empty
    if len(ligand_list) == 0:
        dataset_row['lig type'] = 0
    
    # store information
    dataset_info_row['lig (type)'] = [ligand_list]
    lig_type_dict[mol_name] = ligand_list
    dataset_info_row['lig num (type)'] = len(ligand_list)
    
    return dataset_row, dataset_info_row, lig_info_df, ligand_list


def store_lig_mass(lig, lig_info_df, kwd, val):
    '''
    store type of ligand in df
    
    Args:
        ligands (list): list of ligands to be added
        lig_info_df (pd.DataFrame): see description below
        ...
        ...

    Returns:
        lig_info_df (pd.DataFrame): see description below
    '''
    # check if ligand already available in df
    if lig in lig_info_df['lig id'].values:
        lig_info_df.loc[lig_info_df['lig id'] == lig, kwd] = val
    # add ligand to dataframe
    else:
        lig_info_row = pd.DataFrame(zip([lig], [None], [None], [0], [0], [0], [0], [0], [0], [0]), columns=['lig id', 'mass',
                    'mass status', 'suitable type', 'NMP/NDP/NTP', 'water', 'complex', 'small mol', 'solvent/buffer', 'linker'])
        lig_info_row[kwd] = val
        lig_info_df = lig_info_df.append(lig_info_row)
    return lig_info_df


def calc_ligand_mass(mol, ligands, lig_info_df):
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


def filter_site_rna_fraction(rna, prot, lig):
    '''
    filter ligands by fraction of RNA atoms in binding site
    
    Args: rna, prot, lig (moleculekit molecule objects)
    
    Returns:
        ligand_names_for_res (list): ligand names for all binding site residues in molecule
        ligand_res_list (list): residue numbers for all ligands in molecule
        site_frac_list (list): fraction of RNA atoms within binding site for all ligands in molecule
        keep_res_list (list): residue numbers for all suitable binding site ligands
        ligand_names_for_res_filtered (list): ligand names for all suitable binding site residues
        len(keep_res_list) (int): number of suitable ligands in molecule
    '''
    # reshape rna coordinates for analysis
    rna_coords = reshape_coords(rna.coords)
    prot_coords = reshape_coords(prot.coords)

    # generate list with all ligands
    ligand_res_list = list(set(lig.resid.tolist()))
    # remove negative residue ids
    ligand_res_list = list(filter(lambda x: x >= 0, ligand_res_list))

    # initialize empty list to store site RNA fractions
    site_frac_list = []
    ligand_names_for_res = []
    ligand_names_for_res_filtered = []
    
    for res in ligand_res_list:
        single_lig = get_single_lig(lig, res)
        site_frac = calc_site_frac(single_lig, rna_coords, prot_coords)
        site_frac_list.append(site_frac)
        ligand_names_for_res += list(np.unique(single_lig.resname))

        if site_frac >= BINDING_SITE_FRAC_THRESHOLD:
            keep_res_list.append(res)
            ligand_names_for_res_filtered += list(np.unique(single_lig.resname))
    
    return ligand_names_for_res, ligand_res_list, site_frac_list, keep_res_list, ligand_names_for_res_filtered, len(keep_res_list)
        

def calc_site_frac(single_lig, rna_coords, prot_coords, threshold=BINDING_SITE_FRAC_DIST_THRESHOLD):
    '''
    calculate fraction of RNA atoms of binding site (fraction of RNA to protein atoms in proximity to ligand)

    Args:
        single_lig (moleculekit molecule): molecule object for single ligand that defines binding site
        rna_coords: coordinates of all rna atoms of molecule
        prot_coords: coordinates of all protein atoms of molecule
        threshold (float): RNA fraction for binding site classification --> suitable for training RNA-focused deep learning models?

    Returns:
        fraction of RNA atoms within binding site (float between 0 and 1)
    '''
    # calculate distance to nearest ligand atom for all rna and prot atoms
    single_lig_coords = reshape_coords(single_lig.coords)
    rna_dist = distance.cdist(single_lig_coords, rna_coords, 'euclidean')
    rna_dist = np.amin(rna_dist, axis=0)
    prot_dist = distance.cdist(single_lig_coords, prot_coords, 'euclidean')
    prot_dist = np.amin(prot_dist, axis=0)

    # get number of atoms close to ligand: binding site atoms
    rna_num = np.sum(np.where(rna_dist < threshold, 1, 0))
    prot_num = np.sum(np.where(prot_dist < threshold, 1, 0))

    # calculate fraction of RNA atoms of binding site
    return rna_num / (rna_num + prot_num)


def warn(*args, **kwargs):
    """
    silence warnings
    """
    pass



if __name__ == '__main__':

    # silence warnings / info
    logging.getLogger('moleculekit.molecule').setLevel(logging.ERROR)
    logging.getLogger('moleculekit.readers').setLevel(logging.ERROR)
    logging.getLogger('__name__').setLevel(logging.ERROR)
    warnings.warn = warn

    # load paths to pdb files
    pdb_path = os.path.join(BASE_PATH, 'data/mol/rna/pdb/*.pdb')
    pdb_files = glob.glob(pdb_path)
    
    # load smiles strings for all pdb ligands
    # file "Components-smiles-stereo-oe.smi" can be downloaded from pdb: http://ligand-expo.rcsb.org/ld-download.html
    smiles_path = os.path.join(BASE_PATH, 'data/dataset_preparation/Components-smiles-stereo-oe.smi')
    smiles_df = pd.read_csv(smiles_path, delimiter='\t', header=None, on_bad_lines='warn')

    # define & create directory to save results
    out_path = os.path.join(BASE_PATH, 'data/dataset_preparation/rna')
    os.makedirs(out_path, exist_ok=True)

    # initialize empty dataframes to store results from data processing
    #
    # note: lists will also be stored in dataframes to generate a user-friendly overview over the dataset in a table-like format
    #
    # dataset_df: stores if pdb entry ('pdb id') fullfills filtering criterium (1) or not (0)
    # filtering criteria:
    #  - Fraction of RNA atoms of pdb entry ('rna frac'): specifies if relative fraction of RNA to protein atoms > 0.5
    #  - RNA structure type ('structure type'): specifies if structure type suitable for learning tasks on RNA structures or not
    #  - Ligand types of enclosed ligands ('lig type'): specifies if at least one ligand with suitable ligand type present in pdb entry
    #  - Ligand mass of enclosed ligands ('lig mass'): specifies if at least one ligand with suitable mass (200 - 700 Da) present
    #  - Fraction of RNA atoms at ligand binding sites ('site frac'): specifies if the fraction of RNA atoms (relative fraction of RNA to
    #    protein atoms) of the binding site around at least one ligand if > 0.5
    #  - Grid site of pdb entry ('grid size'): specifies if structure fits in cube with edgelength of 80 Angström (only relevant for
    #    usecase examined in this project not for usage of dataset in general)
    dataset_df = pd.DataFrame(columns=['pdb id', 'rna frac', 'structure type', 'lig type', 'lig mass', 'site frac', 'elements',
                                        'grid size'])
    
    # dataset_info_df: stores information relevant for filtering of pdb entries ('pdb id')
    # information:
    #  - List of all ligand ids (unsuitable and suitable for training): 'unfiltered lig' (duplicate/... ids only listed once)
    #  - List of all metal ids (metal ions) present in structure: 'metal'
    #  - List of ligand ids for all covalently bound ligands: 'covalent lig
    #  - List of all ligands that are not metal ions or covalentely bound: 'lig'
    #  - List of all suitable ligands (suitable type and mass): 'lig filtered'
    #  - Number of suitable ligands (suitable type and mass): 'lig num'
    #  - List of all ligand residue ids (only for ligands with suitable type and mass) in structure: 'res'
    #  - List of corresponding ligand ids for each residue id in 'res': 'res lig'
    #  - List of RNA atom fractions within binding site for each res id in 'res': 'res site frac'
    #  - List of residue ids with suitable RNA atom binding site fraction: 'res filtered'
    #  - List of corresponding ligand ids for residues listed on 'res filtered': 'res filtered lig'
    #  - Number of suitable binding sites: 'res num'
    #  - List of ligands with suitable ligand type: 'lig (type)'
    #  - Number of ligands with suitable type: 'lig num (type)'
    #  - List of ligands with suitable ligand mass: 'lig (mass)'
    #  - Number of ligands with suitable mass: 'lig num (mass)'
    #  - Fraction of RNA atoms within whole molecule: 'rna frac'
    #  - RNA structure type: 1 if molecule has type (column titel), 0 if not; columns 'ribosomal rna' - 'other structure'
    #  - Number of structure type labels == 1 for molecule: 'type num'
    #  - List of elements present in molecule: 'elements'
    #  - List of elements not encoded by RNet but present in molecule: 'elements out'
    #  - Edge length of cube used to describe molecule to train RNet in dimension 1: 'size dim1'
    #  - Edge length of cube used to describe molecule to train RNet in dimension 2: 'size dim2'
    #  - Edge length of cube used to describe molecule to train RNet in dimension 3: 'size dim3'
    #  - Minimal edge length of cube describing molecule: maximal value of 3 cube dimensions described before ('size')
    dataset_info_df = pd.DataFrame(columns=['pdb id', 'unfiltered lig', 'metal', 'covalent lig', 'lig', 'lig filtered', 'lig num',
                                        'res', 'res lig', 'res site frac', 'res filtered', 'res filtered lig', 'res num', 'lig (type)',
                                        'lig num (type)', 'lig (mass)', 'lig num (mass)', 'rna frac', 'ribosomal rna', 'viral rna',
                                        'ribozyme', 'riboswitch', 'aptamer', 'primer complex', 'splicing-related', 'other structure',
                                        'type num', 'elements', 'elements out', 'size dim1', 'size dim2', 'size dim3', 'size'])
    
    # lig_info_df: stores information relevant for assessing suitablility of ligands
    # information:
    #  - 'lig id': residue name of ligand (ids as used in pdb)
    #  - 'mass': molecular mass of ligand in Da
    #  - 'mass status': specifies if molecular mass exact value (exact) or approximation (approximation) or if error during mass
    #    calculation occured (error: if error: mass = 0)
    #  - further colums: specifies how often a ligand is present in all pdb structures having the function/type specified by the column
    #    name (if a ligand is present in a single pdb entry multiple times with a specific function, this contribution is only counted
    #    as 1)
    lig_info_df = pd.DataFrame(columns=['lig id', 'mass', 'mass status', 'suitable lig', 'suitable type', 'NMP/NDP/NTP', 'water',
                                        'complex', 'small mol', 'solvent/buffer', 'linker', 'covalent'])
    
    # initialize dictionaries
    metal_dict = {} # cf. column 'metal' in dataset_info_df
    link_dict = {} # cf. column 'covalent lig' in dataset_info_df
    lig_dict = {} # cf. column 'lig' in dataset_info_df
    lig_filted_dict = {} # cf. column 'lig filtered' in dataset_info_df
    res_frac_dict = {} # cf. column 'res site frac' in dataset_info_df
    res_filtered_dict = {} # cf. column 'res filtered' in dataset_info_df
    res_filtered_lig_dict = {} # cf. column 'res filtered lig' in dataset_info_df
    lig_type_dict = {} # cf. column 'lig (type)' in dataset_info_df
    lig_mass_dict = {} # cf. column 'lig (mass)' in dataset_info_df
    elements_out_dict = {} # cf. column 'elements out' in dataset_info_df

    # number of molecules to be processed
    mol_num = len(pdb_files)

    # loop over molecules for filtering
    for i, file_path in enumerate(pdb_files):

        # ###################################################### 0. load molecule ######################################################
        # load molecule
        mol = Molecule(file_path, validateElements=False)

        # keep single frame
        check_frames(mol)

        # load pdb id of molecule
        mol_name = os.path.basename(os.path.realpath(os.path.splitext(file_path)[0]))

        # logging
        print('loaded molecule', str(mol_name), '|', str(round((i+1)/mol_num*100)), 'percent completed')

        # initialize new rows for dataframes
        dataset_row = pd.DataFrame(zip([''], [1], [1], [1], [1], [1], [1], [1]), columns=['pdb id', 'rna frac', 'structure type',
                                        'lig type', 'lig mass', 'site frac', 'elements', 'grid size'])
        dataset_info_row = pd.DataFrame(columns=['pdb id', 'unfiltered lig', 'metal', 'covalent lig', 'lig', 'lig filtered', 'lig num',
                                        'res', 'res lig', 'res site frac', 'res filtered', 'res filtered lig', 'res num', 'lig (type)',
                                        'lig num (type)', 'lig (mass)', 'lig num (mass)', 'rna frac', 'ribosomal rna', 'viral rna',
                                        'ribozyme', 'riboswitch', 'aptamer', 'primer complex', 'splicing-related', 'other structure',
                                        'type num', 'elements', 'elements out', 'size dim1', 'size dim2', 'size dim3', 'size'])
        dataset_row['pdb id'] = [mol_name]
        dataset_info_row['pdb id'] = [mol_name]

        # ################################# 1. filter structures according to their fraction of RNA atoms ##################################
        # calculate fraction of RNA atoms
        rna_frac = calc_rna_frac(mol)

        # store fraction of RNA atoms
        dataset_info_row['rna frac'] = rna_frac
        
        # sort out pdb entries
        if rna_frac < RNA_FRAC_THRESHOLD:
            dataset_row['rna frac'] = [0]
        
        # ############################ 2. & 3. filter structures according to ligand type & structure type #############################
        # get & store all potential ligands from pdb entry (incl. metal ions, covalent ligands, non-natural building blocks)
        ligands = get_ligands(mol)
        dataset_info_row['unfiltered lig'] = [ligands]

        # text mining of corresponding pdb file
        pdb_path = os.path.join(BASE_PATH, 'data/mol/rna/pdb', str(mol_name) + '.pdb') # get path to pdb file
        type_list, ligands, metal_list, link_list, p_dict, ribo_kwds_list = text_mining_pdb(pdb_path, ligands)

        # store extracted information
        dataset_info_row['lig'] = [ligands]
        lig_dict[mol_name] = ligands
        dataset_info_row['metal'] = [metal_list]
        metal_dict[mol_name] = metal_list
        dataset_info_row['covalent lig'] = [link_list]
        link_dict[mol_name] = link_list
        lig_info_df = store_lig_type(link_list, lig_info_df, 'covalent')
        
        # adjust and filter RNA structure types
        type_list, dataset_row, dataset_info_row = filter_structure_type(type_list, dataset_row, dataset_info_row)
        
        # check if further filtering by ligand type necessary
        if 5 in type_list:
            # sort out ligands of primer-template complexes for pdb entry
            lig_info_df = store_lig_type(ligands, lig_info_df, 'NMP/NDP/NTP')
            suitable_lig_type = []
        else:
            # filter ligands by ligand type
            dataset_row, dataset_info_row, lig_info_df, suitable_lig_type = filter_ligand_type(dataset_row, dataset_info_row,
                    lig_info_df, ligands, link_list, p_dict, ribo_kwds_list)
            lig_info_df = store_lig_type(suitable_lig_type, lig_info_df, 'suitable type')

        # ################################# 4. filter structures according to mol. mass of ligands ###################################
        # calculate mass of all ligands
        lig_info_df = calc_ligand_mass(mol, ligands, lig_info_df)
        
        # get ligands with suitable mass for pbd entry
        suitable_lig_mass = [key for key in ligands if LOWER_LIG_MASS <= lig_info_df.loc[lig_info_df['lig id'] == key, 'mass'].iloc[0]
                <= UPPER_LIG_MASS]
        
        # store results
        dataset_info_row['lig (mass)'] = [suitable_lig_mass]
        lig_mass_dict[mol_name] = suitable_lig_mass
        dataset_info_row['lig num (mass)'] = len(suitable_lig_mass)
        
        # check if at least one ligand with suitable mass, otherwise sort out pdb entry
        if len(suitable_lig_mass) == 0:
            dataset_row['lig mass'] = [0]

        # ######################## 5. filter structures according to fraction of RNA atoms within binding site #########################
        # get suitable ligands from mass calculation and ligand type determination and only determine fraction of RNA atoms within
        # binding site for those ligands to limit the computational complexity
        suitable_ligands = np.intersect1d(suitable_lig_type, suitable_lig_mass)
        keep_res_list = []

        if len(suitable_ligands) > 0:
            # decompose molecule into rna, protein and ligand
            rna, prot, lig = decompose_mol(mol, suitable_ligands, link_list)

            # filter ligands (here: occurances of ligand)
            lig_names, res_list, res_frac, res_filtered, lig_names_filtered, res_num = filter_site_rna_fraction(rna, prot, lig)
        
        else:
            lig = None

            # define RNA
            rna = mol.copy()
            link_codes = link_list + RNA_CODES
            rna.filter('resname '+' '.join([str(i) for i in link_codes]))

            lig_names, res_list, res_frac, res_filtered, lig_names_filtered, res_num = [], [], [], [], [], 0

        # store results
        dataset_info_row['res'] = [res_list]
        dataset_info_row['res lig'] = [lig_names]
        dataset_info_row['res site frac'] = [res_frac]
        res_frac_dict[mol_name] = res_frac
        dataset_info_row['res filtered'] = [res_filtered]
        res_filtered_dict[mol_name] = res_filtered
        dataset_info_row['res filtered lig'] = [lig_names_filtered]
        res_filtered_lig_dict[mol_name] = lig_names_filtered
        dataset_info_row['res num'] = res_num
            
        # check if at least one ligand occurance comes with suitable fraction of RNA atoms in binding site, otherwise sort out pdb entry
        if res_num == 0:
            dataset_row['site frac'] = [0]
        
        # get suitable ligands after RNA fraction filter
        suitable_ligands = np.unique(lig_names_filtered)

        # store suitable ligands
        dataset_info_row['lig filtered'] = [suitable_ligands]
        lig_filted_dict[mol_name] = suitable_ligands
        dataset_info_row['lig num'] = len(suitable_ligands)
        lig_info_df = store_lig_type(suitable_ligands, lig_info_df, 'suitable lig')

        # ####################################### 6. filter structures according to element occurrance ##################################
        elements = np.unique(rna.element)
        dataset_info_row['elements'] = [elements]
        elements = np.setdiff1d(elements, ELEMENTS_CH + ['H'])
        dataset_info_row['elements out'] = [elements]
        elements_out_dict[mol_name] = elements
        if len(elements) > 0:
            dataset_row['elements'] = [0]
        
        # ######################################## 7. filter structures according to grid size ##########################################
        # get bounding box of molecule
        bbox = boundingBox(mol)

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
        dataset_df.to_csv(os.path.join(out_path, 'rna_dataset_tmp.csv'), sep=',', header=True, index=False)
        dataset_info_df.to_csv(os.path.join(out_path, 'rna_dataset_info_tmp.csv'), sep=',', header=True, index=False)
        lig_info_df.to_csv(os.path.join(out_path, 'lig_dataset_info_tmp.csv'), sep=',', header=True, index=False)


    # save all results

    # save dictionaries
    result_dict = {
        'metal': metal_dict,
        'link': link_dict,
        'lig': lig_dict,
        'lig_filtered': lig_filted_dict,
        'res_frac': res_frac_dict,
        'res_filtered': res_filtered_dict,
        'res_filtered_lig': res_filtered_lig_dict,
        'lig_type': lig_type_dict,
        'lig_mass': lig_mass_dict,
        'elements_out': elements_out_dict
    }
    # save to pkl file
    pickle_save(os.path.join(out_path, 'rna_result_dicts.pkl'), result_dict)

    # save dataframes
    dataset_df.to_csv(os.path.join(out_path, 'rna_dataset_raw.csv'), sep=',', header=True, index=False)
    dataset_info_df.to_csv(os.path.join(out_path, 'rna_dataset_info_raw.csv'), sep=',', header=True, index=False)
    lig_info_df.to_csv(os.path.join(out_path, 'lig_dataset_info.csv'), sep=',', header=True, index=False)

    dataset_info_df = dataset_info_df[dataset_info_df['rna frac'] >= 0.5]
    data_ids = dataset_info_df['pdb id'].values
    dataset_df = dataset_df[dataset_df['pdb id'].isin(data_ids)]
    dataset_df = dataset_df.astype({'rna frac': int, 'structure type': int, 'lig type': int, 'lig mass': int, 'site frac': int,
                                   'elements': int, 'grid size': int})
    dataset_df.to_csv(os.path.join(out_path, 'rna_dataset.csv'), sep=',', header=True, index=False)
    dataset_info_df.to_csv(os.path.join(out_path, 'rna_dataset_info.csv'), sep=',', header=True, index=False)
    
    # filter for suitable pdb entries
    # 1. pdb entries fullfilling all criteria except grid-size, element filter
    dataset_df = dataset_df.astype({'rna frac': int, 'structure type': int, 'lig type': int, 'lig mass': int, 'site frac': int,
                                   'elements': int, 'grid size': int})
    dataset_df['used for RNet'] = dataset_df.iloc[:,1:6].sum(axis=1)
    dataset_df_f1 = dataset_df[dataset_df['used for RNet'] >= 5]
    dataset_df['used for RNet'] = np.where(dataset_df.iloc[:,1:8].sum(axis=1) >= 7, 1, 0)
    dataset_df_f1.to_csv(os.path.join(out_path, 'rna_dataset_filter1.csv'), sep=',', header=True, index=False)

    # 2. pdb entries fullfilling all criteria (only structures fitting in box with specified size)
    dataset_df['filter sum'] = dataset_df.iloc[:,1:8].sum(axis=1)
    dataset_df_f2 = dataset_df[dataset_df['filter sum'] >= 7]
    dataset_df_f2.to_csv(os.path.join(out_path, 'rna_dataset_filter2.csv'), sep=',', header=True, index=False)
