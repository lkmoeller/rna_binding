import os, argparse, qml
import numpy as np
import pandas as pd
from qml.representations import *
from qml.kernels import laplacian_kernel
from qml.representations import get_slatm_mbtypes
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from rna_binding.utils import BASE_PATH, pickle_load, pickle_save



# define rna types
RNA_TYPES = ['ribosomal rna', 'viral rna', 'ribozyme', 'riboswitch', 'aptamer', 'primer complex', 'splicing-related', 'other structure']



def generate_qml_compounds(dataset_info, dataset_dict):
    """ Function to bring data in right format for SLATM descriptor generation. Takes dataframe (dataset_info) and dictionary
        (dataset_dict) from rna_data_prep.py as input and returns lists with PDB ids (mol_names), RNA structure type (mol_types)
        Ligands enclosed in binding sites used for training RNet (mol_lig), QML compounds used for SLATM generation (mols).
    """
    # initialize empty lists
    mol_names, mol_types, mol_lig, mols = [], [], [], []

    for i in RNA_TYPES:
        mol_per_type = dataset_info.loc[dataset_info[i] == 1, 'pdb id'].values
        for mol_id in mol_per_type:
            if mol_id not in mol_names:
                mol_names.append(mol_id)
                mol_types.append([i])
                mol_lig.append(list(dataset_dict['lig_filtered'][mol_id]))
                mol = qml.Compound()
                mol.read_xyz(os.path.join(xyz_path, mol_id + '.xyz'))
                mols.append(mol)
            else: # multiple type annotations
                mol_index = mol_names.index(mol_id)
                mol_types[mol_index].append(i)
    
    return mol_names, mol_types, mol_lig, mols


def generate_slatm_representation(mols):
    """ Generate SLATM representation for all molecules in mols (list of QML compound objects) and return array with corresponding
        representations.
    """
    # get molecular charges for slatm calculation and append to qml molecule objects
    mbtypes = get_slatm_mbtypes([mol.nuclear_charges for mol in mols])

    # generate slatm representation for qml molecule objects
    print("\n -> generate slatm")
    for mol in tqdm(mols):
        mol.generate_slatm(mbtypes, local=False)
    
    # get and return array with slatm representations
    return np.asarray([mol.representation for mol in mols])


def perform_pca(X, num=None):
    """ Perform PCA for matrix X (quadratic matrix based on pairwise binding site similarities) and num PCs.
        Return transformed matrix X, variance explained by each PC (np.array), total variance explained by transformed X (float, for
        num=None value is 1).
    """
    pca = PCA(n_components=num)
    pca.fit(X)
    expl_variance = pca.explained_variance_ratio_
    X = pca.transform(X)
    return X, expl_variance, np.sum(expl_variance)


def get_sim_matrix(SLATM_array, sigma, pca_comp=None):
    """ Calculates pairwise similarities between all binding sites.
        Uses SLATM representations (SLATM_array, np.array, len: d) and sigma parameter as input for Laplacian.
        Returns transformed Matrix X (np.array, size: d x d) after Laplacian Kernel calculation and PCA.
    """
    SIM_matrix = laplacian_kernel(SLATM_array, SLATM_array, sigma)
    SIM_matrix, expl_var, expl_var_sum = perform_pca(SIM_matrix, pca_comp)
    return SIM_matrix, expl_var, expl_var_sum


def optimize_cnum(SLATM_array, sigma):
    """ Optimizes number of clusters for given SLATM_array and sigma parameter so that mean silhouette coefficient maximal.
        Returns optimal cluster number, silhouette coefficient of corresponding cluster number, array of tested cluster numbers,
        array of mean silhouette coefficients for all tested cluster numbers.
    """
    cluster_nums = [i for i in range(2, 31)]
    min_cluster_num = 9 # corresponds to 10 clusters since 2 first elements in array
    silhouette = []

    SIM_matrix, _, _ = get_sim_matrix(SLATM_array, sigma)

    for cluster_num in tqdm(cluster_nums):
        clustering = AgglomerativeClustering(linkage='ward', n_clusters=cluster_num)
        clustering.fit(SIM_matrix)
        silhouette_avg = silhouette_score(SIM_matrix, clustering.labels_)
        silhouette.append(silhouette_avg)
    
    best_cnum_index = np.argmax(silhouette[min_cluster_num:]) + min_cluster_num

    return cluster_nums[best_cnum_index], silhouette[best_cnum_index], cluster_nums, silhouette


def optimize_sigma(SLATM_array):
    """
        Optimizes sigma parameter for Laplacian kernel for given SLATM_array
    """
    sigmas = [0.2*2**i for i in range(2, 16)]
    sigma_silhouette_list, sigma_cluster_list  = [], []

    for sigma in sigmas:
        print(' -> evaluating sigma', sigma)
        best_cluster, max_silhouette, _, _ = optimize_cnum(SLATM_array, sigma)
        sigma_silhouette_list.append(max_silhouette)
        sigma_cluster_list.append(best_cluster)
        print('highest silhouette coefficient', max_silhouette, 'for sigma', sigma, 'and cluster number', best_cluster)
    
    best_sigma_index = np.argmax(sigma_silhouette_list)

    return sigmas[best_sigma_index], sigma_cluster_list[best_sigma_index], sigmas, sigma_cluster_list, sigma_silhouette_list



if __name__ == '__main__':
    # argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("-filter", type=int, required=False, default=2, help='')
    parser.add_argument("-optimize", type=int, required=False, default=1, help='')
    parser.add_argument("-sigma", type=int, required=False, default=6554, help='')
    parser.add_argument("-cnum", type=int, required=False, default=6, help='')
    args = parser.parse_args()

    # paths to pdb files
    xyz_path = os.path.join(BASE_PATH, 'data/mol/rna/xyz')

    # set output path
    save_path = os.path.join(BASE_PATH, 'data/dataset_preparation/rna')

    # load list of rna molecules and stored information about rna type etc.
    dataset_dict = pickle_load(os.path.join(BASE_PATH, 'data/dataset_preparation/rna/rna_result_dicts.pkl'))
    dataset_df = pd.read_csv(os.path.join(BASE_PATH, 'data/dataset_preparation/rna/rna_dataset_filter' + str(args.filter) + '.csv'), delimiter=',', index_col=None) 
    dataset_info_df = pd.read_csv(os.path.join(BASE_PATH, 'data/dataset_preparation/rna/rna_dataset_info.csv'), delimiter=',', index_col=None) 

    # get pdb ids of interest
    mol_ids = dataset_df['pdb id'].values

    # filter information dataset
    dataset_info_df = dataset_info_df[dataset_info_df['pdb id'].isin(mol_ids)]

    # generate qml compounds & process information about mols
    mol_names, mol_types, mol_lig, mols = generate_qml_compounds(dataset_info_df, dataset_dict)

    # generate slatm representation for molecules
    SLATM_array = generate_slatm_representation(mols)

    # get sigma & cluster number
    if int(args.optimize) == 1:
        sigma, _, sigmas, sigma_cluster_list, sigma_silhouette_list = optimize_sigma(SLATM_array)
        sigma = round(sigma)
        cluster_num, best_av_sil, cluster_nums, silhouette = optimize_cnum(SLATM_array, sigma)
        print(' -> found best parameters: sigma', sigma, 'cluster number', cluster_num)
    else:
        sigma, cluster_num = args.sigma, args.cnum
    
    # do final clustering
    SIM_matrix, _, _ = get_sim_matrix(SLATM_array, sigma)
    clustering = AgglomerativeClustering(linkage='ward', n_clusters=cluster_num)
    clustering.fit(SIM_matrix)
    single_silhouette_values = silhouette_samples(SIM_matrix, clustering.labels_)
    
    # save cluster labels for data splitting
    label_dict = {}
    for i in range(cluster_num):
        mols_per_cluster = np.where(clustering.labels_ == i)
        label_dict[i] = np.array(mol_names)[mols_per_cluster].tolist()
    pickle_save(os.path.join(save_path, 'rna_cluster_labels_' + str(args.filter) + '_' + str(sigma) + '_' + str(cluster_num) + '.pkl'), label_dict)

    # save information for plotting
    if int(args.optimize) == 1:
        SIM_matrix_plot, expl_var, expl_var_sum = get_sim_matrix(SLATM_array, sigma, 2)
        info_dict = {
            'coordinates_plot': SIM_matrix_plot,
            'coordinates_full': SIM_matrix,
            'expl_var_plot': expl_var,
            'expl_var_sum_plot': expl_var_sum,
            'mol_names': mol_names,
            'mol_types': mol_types,
            'mol_lig': mol_lig,
            'sigmas_opt': sigmas,
            'sigma_cluster_list_opt': sigma_cluster_list,
            'sigma_silhouette_list_opt': sigma_silhouette_list,
            'cluster_nums_best_sigma': cluster_nums,
            'silhouette_best_sigma': silhouette,
            'single_silhouette_best_cnum': single_silhouette_values,
            'cluster_labels_best_cnum': clustering.labels_,
            'av_silhouette_best_cnum': best_av_sil,
            'cluster_num': cluster_num,
            'sigma': sigma
        }
        pickle_save(os.path.join(save_path, 'rna_cluster_info_' + str(args.filter) + '_' + str(sigma) + '_' + str(cluster_num) + '.pkl'), info_dict)
