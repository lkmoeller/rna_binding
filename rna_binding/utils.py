"""
Utils for rna_binding project:
Please import util functions into other project files using rna_binding.utils(function)
Please check if PYTHONPATH is set to module rna_binding before using rna_binding project

Authors: Lukas Moeller
Date: 01/2022
"""



import pickle, os, torch
import numpy as np
from moleculekit.molecule import Molecule



# nucleobases constituting RNA
RNA_CODES=['G','C','U','A']
# protein residues
PROT_CODES=['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR']
# elements of channels used to train RNet
ELEMENTS_CH = ["C", "N", "O", "F", "P", "S", "Br", "I"]
# elements that potential ligands could contain
ELEMENTS_LIG = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
# atomic mass of elements in amu (atomic mass unit) or Da (Dalton)
ELEMENT_MASS = {"H": 1.00797, "C": 12.011, "N": 14.0067, "O": 15.9994, "F": 18.998403, "P": 30.97376, "S": 32.06, "Cl": 35.453, "Br": 79.904,  "I": 126.9045}
# Edge length of grids used to describe molecules for training RNet
BOX_SIZE = 80
# Define distance to classify atoms as binding site (same distance than used for scPDB database: Kellenberger et. al, J. Chem. Inf. a.
# Mod. 46, 717–727 (2006)), in Angstrom
SITE_DIST = 6.5
# Define resolution descrease factor of output compared to input grids
RES_DECREASE = 4
# Define size of single voxel edge length (input grids) in Angstrom
VOXEL_SIZE = 1
# Dictionary to generate paths to rna/protein directories
MODE_DICT = {0: 'rna', 1: 'prot'}
# Dictionary to get correct names for rna/protein directories
MODE_DICT_PDB = {0: 'pdb', 1: 'scpdb'}
# Define center of molecule after centering
MOL_CENTER = [0, 0, 0]
# minimum mass of ligands included in filtered dataset (in Da)
LOWER_LIG_MASS = 200
# maximal mass of ligands included in filtered dataset (in Da)
UPPER_LIG_MASS = 700


# Get base path to working directory
BASE_PATH = os.path.dirname(os.path.realpath(__file__))

# Specify if GPU resources available for training/prediction
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def print_model_info(model):
    """
    Print information about model (torch) before start of training/prediction
    """
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    sum_params = sum([np.prod(e.size()) for e in model_parameters])
    print("\nModel architecture: ", model)
    print("\nNum model parameters: ", sum_params)
    print("\nLoop over model-weights:")
    for name, e in model.state_dict().items():
        print(name, e.shape)


def write_xyz(mol_atoms, mol_coords, out_path):
    """
    generate xyz file for molecule with the elements specified in 'mol_atoms' and coordinates specified in 'mol_coords',
    store xyz file after generation with the file name (in the directory) 'out_path'
    """
    # save mol atoms and coordinates to xyz file
    with open(out_path, 'w') as f_out:
        f_out.write(str(len(mol_atoms)) + '\n\n')
        for a, (x,y,z) in zip(mol_atoms, mol_coords.squeeze()):
            f_out.write(str(a)+'\t'+str(x)+'\t'+str(y)+'\t'+str(z)+'\n')


def load_mol(mol_path):
    """ Returns moleculekit.Molecule object from pdb-path

    Args:
        mol_path: path to pdb-file of interest

    Returns:
        mol (moleculekit.Molecule): molecule of interest
    """
    return Molecule(mol_path)


def geom_center(mol):
    """ Returns geometric center of molecule

    Args:
        mol (moleculekit.Molecule): molecule

    Returns:
        np.array: geometric center of mol
    """
    return np.mean(np.squeeze(mol.coords), axis=0)


def center_mol(mol_coords, center):
    """ Centers molecule to given center

    Args:
        mol_coords (np.array): xyz-coords of molecule
        center (np.array): array defining center of interest
    
    Returns:
        mol_coords (np.array): centered xyz-coords of molecule
    """
    return mol_coords - np.atleast_3d(center)


def check_frames(mol):
    """
    check number of frames of moleculekit object and return only first frame
    """
    if mol.numFrames > 1:
        mol.dropFrames(keep=[0])


def reshape_coords(coords):
    """
    reshape coordinates to len(coordinates) x 3 array to simplify distance calculations
    """
    dim_0, dim_1 = len(coords), 3
    return coords.reshape((dim_0, dim_1))


def reshape_coords_back(coords):
    """
    reshape coordinates back to original shape (as in moleculekit objects) after reshaping with 'reshape_coords'
    """
    dim_0, dim_1 = len(coords), 3
    return coords.reshape((dim_0, dim_1, 1))


def pickle_load(path_2_pkl):
    """
    load pkl object that can be found at the path path_2_pkl
    """
    with open(path_2_pkl, "rb") as fp:
        pkl  = pickle.load(fp)
    return pkl


def pickle_save(path, obj):
    """
    save the objects 'obj' in the directory 'path' as pkl object
    """
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def decompose_mol(mol, lig_codes, link_codes, metal_codes=[]):
    """Decomposes Molecule object in separate objects for RNA, Protein residues and Ligands
    
    Args:
        mol (moleculekit.Molecule): molecule of interest
        lig_codes (list): python list with pdb codes for ligands that are enclosed in binding site that should be returned
        link_codes (list): python list with pdb codes for covalently bound ligands
        metal_codes (list): python list with pdb codes for metal residues
    
    Returns:
        rna, prot, lig (moleculekit.Molecule)
    """

    # define ligands
    lig = mol.copy()
    lig.filter('resname '+' '.join([str(i) for i in lig_codes]))

    # define protein residues
    prot = mol.copy()
    prot.filter('protein')
    
    # define RNA
    rna = mol.copy()
    link_codes = link_codes + metal_codes + RNA_CODES
    rna.filter('resname '+' '.join([str(i) for i in link_codes]))

    return rna, prot, lig


def get_single_lig(lig, res):
    """
    get single ligand from all ligands
    
    Args:
        lig (moleculekit.Molecule): all ligands as moleculekit object
        res (int): residue id of single ligand that should be selected
    
    Returns:
        single_lig (moleculekit.Molecule): selected ligand as moleculekit object
    """
    single_lig = lig.copy()
    single_lig.filter('resid ' + str(res))
    return single_lig



if __name__ == "__main__":
    pass
