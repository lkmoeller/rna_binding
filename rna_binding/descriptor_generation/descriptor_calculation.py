"""
Generate Descriptors for Training and Model Evaluation: Calculation and storage of voxelized grids

Author: Lukas Moeller
Date: 01/2022
"""



import argparse, logging, os
import pandas as pd
import numpy as np
from moleculekit.molecule import Molecule
from scipy.spatial import distance
from rna_binding.utils import (
    check_frames,
    center_mol,
    pickle_save,
    reshape_coords,
    pickle_load,
    get_single_lig,
    BASE_PATH,
    MODE_DICT,
    SITE_DIST,
    RNA_CODES,
    MODE_DICT_PDB
)
from rna_binding.descriptor_generation.voxelize import voxelize, voxelize_save



def decompose_mol(mol, lig_codes, link_codes, metal_codes=[], res_codes=None):
    """ Decomposes Molecule object in separate objects for RNA, Protein residues and Ligands
        Note: function is different from function in utils!
    
    Args:
        mol (moleculekit.Molecule): molecule of interest
        lig_codes (list): python list with pdb codes for ligands that are enclosed in binding site that should be returned
        link_codes (list): python list with pdb codes for covalently bound ligands
        metal_codes (list): python list with pdb codes for metal residues
        res_codes (list): python list with residue ids of all ligands that are enclosed in binding site that should be returned
    
    Returns:
        rna, prot, lig (moleculekit.Molecule)
    """

    # define ligands
    lig = mol.copy()
    lig.filter('resname '+' '.join([str(i) for i in lig_codes]))
    if res_codes is not None:
        lig.filter('resid '+' '.join([str(i) for i in res_codes]))

    # define protein residues
    prot = mol.copy()
    prot.filter('protein')
    
    # define RNA
    rna = mol.copy()
    link_codes = link_codes + metal_codes + RNA_CODES
    rna.filter('resname '+' '.join([str(i) for i in link_codes]))

    return rna, prot, lig


def get_site_atoms(single_lig, rna_coords, site_index_list):
    """ Function to extract indices of all atoms that consitute binding site for corresponding ligand

    Args:
        single_lig (moleculekit object): ligand of interest
        rna_coords (np.array): coordinates of all rna atoms in reshaped form
        site_index_list (list): list with indices of all rna atoms of all (selected) binding sites within PDB entry (new site added
        every time function called until all sites added)

    Returns:
        site_index_list (list): list with indices of all rna atoms of all (selected) binding sites within PDB entry (new site added
        every time function called until all sites added)
    """
    # calculate distance to nearest ligand atom for all rna atoms
    single_lig_coords = reshape_coords(single_lig.coords)
    atom_dist = distance.cdist(single_lig_coords, rna_coords, 'euclidean')
    atom_dist = np.amin(atom_dist, axis=0)

    # get indices of atoms that are within site_dist of ligand
    site_indices = np.asarray(atom_dist < SITE_DIST).nonzero()[0]
    site_index_list += site_indices.tolist()    
    return site_index_list


def get_total_site(rna, prot, lig, analyze_prot=0):
    """ Function to select and generate binding site used for clustering
    
    Args:
        rna, prot, lig (moleculekit objects): rna and ligand objects
        analyze_prot (int): 1: proteins will be analyzed
    
    Returns:
        site_rna, site_prot (moleculekit object): selected binding site
    """
    # generate list with all ligands
    ligand_res_list = set(lig.resid.tolist())
    
    if rna.numAtoms > 0:
        # reshape rna coordinates for analysis
        rna_coords = reshape_coords(rna.coords)

        # search for rna building up binding site
        site_index_list = []
        for res in ligand_res_list:
            single_lig = get_single_lig(lig, res)
            site_index_list = get_site_atoms(single_lig, rna_coords, site_index_list)

        # get combined site objects containing all ligands
        if len(site_index_list) > 0:
            site_rna = rna.copy()
            site_rna.filter('index ' + ' '.join([str(index) for index in list(set(site_index_list))]))
        else:
            site_rna = Molecule().empty(0)
    else:
        site_rna = Molecule().empty(0)
    
    if prot.numAtoms > 0 and analyze_prot == 1:
        # reshape rna coordinates for analysis
        prot_coords = reshape_coords(prot.coords)

        # search for rna building up binding site
        site_index_list = []
        for res in ligand_res_list:
            single_lig = get_single_lig(lig, res)
            site_index_list = get_site_atoms(single_lig, prot_coords, site_index_list)

        # get combined site objects containing all ligands
        if len(site_index_list) > 0:
            site_prot = prot.copy()
            site_prot.filter('index ' + ' '.join([str(index) for index in list(set(site_index_list))]))
        else:
            site_prot = Molecule().empty(0)
    else:
        site_prot = Molecule().empty(0)

    return site_rna, site_prot


if __name__ == '__main__':
    # bsub -W 4:00 -R "rusage[mem=10240]" -n 1 -sp 90 python descriptor_generation/descriptor_calculation.py -mode 0 -path data/crossval/rna/visualization.csv

    # silence moleculekit
    logging.getLogger('moleculekit.molecule').setLevel(logging.ERROR)

    # argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", type=int, required=False, default=0, help='running mode: 0 = RNA, 1 = Protein')
    parser.add_argument("-path", type=str, required=False, default='data/mol/rna/voxel/multiprocessing/vis_sub1.txt')
    parser.add_argument("-use_psite", type=int, required=False, default=1)
    parser.add_argument("-recalc", type=int, required=False, default=1)
    parser.add_argument("-save_vis", type=int, required=False, default=1)
    args = parser.parse_args()

    # load molecules for which binding site should be predicted
    mol_dict = pd.read_csv(os.path.join(BASE_PATH, args.path), header=None)
    mol_names = mol_dict[0].tolist()

    # paths to pdb files
    pdb_path = os.path.join(BASE_PATH, 'data/mol/', str(MODE_DICT[args.mode]), str(MODE_DICT_PDB[args.mode]))

    # path to output files
    save_path = os.path.join(BASE_PATH, 'data/mol', str(MODE_DICT[args.mode]), 'voxel')
    os.makedirs(save_path, exist_ok=True)

    # get list of rna molecules
    dataset_dict = pickle_load(os.path.join(BASE_PATH, 'data/dataset_preparation', str(MODE_DICT[args.mode]), str(MODE_DICT[args.mode]) + '_result_dicts.pkl'))
    
    if args.mode == 0:
        metal_dict = dataset_dict['metal']
        link_dict = dataset_dict['link']
        ligand_dict = dataset_dict['lig_filtered']
        res_dict = dataset_dict['res_filtered']

    # loop over molecules
    for mol_id in mol_names:
        # check if grid already available
        out_path = os.path.join(save_path, str(mol_id) + '.pkl')
        if args.recalc == 0 and os.path.exists(out_path) == True:
            continue

        if args.mode == 0: # RNA
            # load molecule
            mol = Molecule(os.path.join(pdb_path, mol_id + '.pdb'))
            check_frames(mol)

            # get binding sites site
            rna, prot, lig = decompose_mol(mol, ligand_dict[mol_id], link_dict[mol_id], res_codes=res_dict[mol_id])
            site_rna, site_prot = get_total_site(rna, prot, lig, analyze_prot=args.use_psite)
            
            if args.use_psite == 1 and len(site_prot.element) > 0 and len(prot.element) > 0:
                site_atoms = np.concatenate((site_rna.element, site_prot.element))
                site_coords = np.concatenate((site_rna.coords, site_prot.coords))
                mol_atoms = np.concatenate((rna.element, prot.element))
                mol_coords = np.concatenate((rna.coords, prot.coords))
                lig_atoms = lig.element
                lig_coords = lig.coords
            else:
                site_atoms = site_rna.element
                site_coords = site_rna.coords
                mol_atoms = rna.element
                mol_coords = rna.coords
                lig_atoms = lig.element
                lig_coords = lig.coords

        elif args.mode == 1: # Proteins
            # load molecule and site
            path_2_mol = os.path.join(pdb_path, mol_id)
            mol = Molecule(os.path.join(path_2_mol, 'protein.mol2'))
            site = Molecule(os.path.join(path_2_mol, 'site.mol2'))
            lig = Molecule(os.path.join(path_2_mol, 'ligand.mol2'))
            
            check_frames(mol)
            check_frames(site)
            check_frames(lig)

            site_atoms = site.element
            site_coords = site.coords
            mol_atoms = mol.element
            mol_coords = mol.coords
            lig_atoms = lig.element
            lig_coords = lig.coords
        
        # get geometric center of molecule
        center = np.mean(np.squeeze(mol_coords), axis=0)

        # center molecule
        mol_coords = center_mol(mol_coords, center)
        site_coords = center_mol(site_coords, center)
        lig_coords = center_mol(lig_coords, center)

        # voxelize
        if args.save_vis == 0:
            tensor_mol, tensor_target = voxelize(mol_atoms, mol_coords, site_atoms=site_atoms, site_coords=site_coords)
        else:
            single_save = os.path.join(save_path, str(mol_id))
            os.makedirs(single_save, exist_ok=True)
            tensor_mol, tensor_target = voxelize_save(mol_atoms, mol_coords, site_atoms=site_atoms, site_coords=site_coords, lig_atoms=lig_atoms, lig_coords=lig_coords, save_path=single_save)
    
        # save molecule grids
        pickle_save(out_path, (tensor_mol, tensor_target))
        
        # logging
        print(mol_id + ' finished.')
    