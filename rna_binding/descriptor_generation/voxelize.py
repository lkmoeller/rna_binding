"""
Script for voxelization of molecule grid structures

Authors: Lukas Moeller, Lorenzo Guerci
Date: 01/2022
"""


import os
import numpy as np
from moleculekit.tools.voxeldescriptors import getVoxelDescriptors
from moleculekit.periodictable import periodictable
from rna_binding.utils import (
    BOX_SIZE,
    RES_DECREASE,
    MOL_CENTER,
    VOXEL_SIZE,
    ELEMENTS_CH,
    write_xyz
)



# number of channels molecule
CH_NUM_MOL = np.array([len(ELEMENTS_CH)])
# VdW Radii for elements of channels
RADII_CH = np.array([periodictable[element].vdw_radius for element in ELEMENTS_CH])  
# Define 3D Size of Grids used to generate voxelized representation for all molecules
CUBE_BOX = [BOX_SIZE, BOX_SIZE, BOX_SIZE]
# Define threshold that contributions from all atoms within voxel must exceed to be considered as binding site voxel (only relevant
# for voxels in proximity of ligands), ensures that empty/close-to-empty voxels are not classified as binding site
VOXELIZATION_THRESHOLD = 0.1



def save_visualizations(tensor_mol_summed, tensor_site_summed, tensor_target, save_path, box_size=CUBE_BOX, voxel_size=VOXEL_SIZE, center=MOL_CENTER, res_dec=RES_DECREASE, **kwargs):
    """
    save visualizations (cube files and xyz files) for tensors specified by tensor_mol_summed, tensor_site_summed, tensor_target
    save_path: directory where visualizations will be saved

    specify elements, coordinates of ligands, molecule, site for xyz file with keyword arguments
    """

    from moleculekit.tools.voxeldescriptors import getCenters
    from moleculekit.util import writeVoxels

    # get 3D vector denoting the minimal corner of the grid
    usercenters, _ = getCenters(boxsize=box_size, center=center, voxelsize=voxel_size)
    min_corner = usercenters.min(axis=0) - np.array([voxel_size / 2] * 3)

    # store objects
    writeVoxels(tensor_mol_summed, os.path.join(save_path, 'mol.cube'), min_corner, [voxel_size, voxel_size, voxel_size])

    # get 3D vector denoting the minimal corner of the grid
    usercenters, _ = getCenters(boxsize=box_size, center=center, voxelsize=voxel_size*res_dec)
    min_corner = usercenters.min(axis=0) - np.array([voxel_size*res_dec / 2] * 3)

    #store objects
    writeVoxels(tensor_site_summed, os.path.join(save_path, 'site.cube'), min_corner, [voxel_size*res_dec, voxel_size*res_dec, voxel_size*res_dec])
    writeVoxels(tensor_target, os.path.join(save_path, 'target.cube'), min_corner, [voxel_size*res_dec, voxel_size*res_dec, voxel_size*res_dec])
    
    # generate xyz-files for ligands and molecule
    for key, value in kwargs.items():
        write_xyz(value[0], value[1], os.path.join(save_path, str(key) + '.xyz'))


def get_atom_channels(mol_atoms):
    """
    generate matrix representing channels for each atom of molecule (mol_atoms) and return generated matrix
    """
    atom_selections = np.concatenate([np.where(mol_atoms == element, 1, 0).reshape(len(mol_atoms), -1) for element in ELEMENTS_CH], dtype=int, axis=1)
    atom_channels = (atom_selections * RADII_CH).astype(np.float32)
    return atom_channels


def channel_summed(tensor):
    """
    summarize contributions of all channels to single value
    """
    input_tensor_total = np.zeros(tensor[0].shape)   
    for i in range(len(ELEMENTS_CH)):
        input_tensor_total += tensor[i]
    return input_tensor_total


def get_tensor(mol_atoms, mol_coords, center, box_size=CUBE_BOX, voxel_size=VOXEL_SIZE):
    """
    get voxelized grid representation (tensor) of size box_size and with voxel size voxel_size for molecule defined by atoms (mol_atoms, array), and atom coordinates (mol_coords, array)
    """
    channels = get_atom_channels(mol_atoms)
    features, _, num_centers = getVoxelDescriptors(mol=None, boxsize=box_size, voxelsize=voxel_size, center=center, userchannels=channels, usercoords=mol_coords, validitychecks=False) 
    tensor = np.transpose(features.reshape(np.concatenate((num_centers, CH_NUM_MOL))), axes=(3, 0, 1, 2)).astype(np.float32)
    return tensor


def define_site(site, threshold=VOXELIZATION_THRESHOLD):
    """
    get binding site for given site tensor: sorts out voxels with low site contribution/empty space
    """
    return np.where(site > threshold, 1, 0)


def voxelize(mol_atoms, mol_coords, site_atoms=None, site_coords=None, center=MOL_CENTER, box_size=CUBE_BOX, voxel_size=VOXEL_SIZE, threshold=VOXELIZATION_THRESHOLD, res_decrease=RES_DECREASE):
    """
    generate voxelized grid prepresentation for molecule
    """
    tensor_mol = get_tensor(mol_atoms, mol_coords, center, box_size, voxel_size)
    tensor_site = get_tensor(site_atoms, site_coords, center, box_size, voxel_size*res_decrease)
    tensor_site_summed = channel_summed(tensor_site)
    tensor_target = define_site(tensor_site_summed, threshold)
    
    return tensor_mol, tensor_target


def voxelize_save(mol_atoms, mol_coords, site_atoms=None, site_coords=None, lig_atoms=None, lig_coords=None, center=MOL_CENTER, save_path=None, box_size=CUBE_BOX, voxel_size=VOXEL_SIZE, threshold=VOXELIZATION_THRESHOLD, res_decrease=RES_DECREASE):
    """
    generate voxelized grid prepresentation for molecule and save visualizations
    """
    tensor_mol = get_tensor(mol_atoms, mol_coords, center, box_size, voxel_size)
    tensor_mol_summed = channel_summed(tensor_mol)
    tensor_site = get_tensor(site_atoms, site_coords, center, box_size, voxel_size*res_decrease)
    tensor_site_summed = channel_summed(tensor_site)
    tensor_target = define_site(tensor_site_summed, threshold)
    
    save_visualizations(tensor_mol_summed, tensor_site_summed, tensor_target, save_path, box_size, voxel_size, center, res_decrease, mol=[mol_atoms, mol_coords], site=[site_atoms, site_coords], lig=[lig_atoms, lig_coords])
        
    return tensor_mol, tensor_target



if __name__ == "__main__":
    pass
