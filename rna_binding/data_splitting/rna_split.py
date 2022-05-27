import os, random, itertools
import numpy as np
from rna_binding.utils import BASE_PATH, pickle_load



# number of folds to be used
folds = 7

# define folds based on clustering, rna type annotations, balanced number of samples per fold
fold_dict = {1: [8, 12], # 25 // - 3
            2: [1, 7], # 26 // - 2
            3: [10], # 35 // + 7
            4: [2], # 27 // - 1
            5: [0, 4], # 28 // 0
            6: [3, 5, 9, 11, 14], # 28 // + 0
            7: [6, 13]} # 26 // - 2

# define list of fold_dict keys available for validation
validation_keys = list(fold_dict.keys())



def print_csv(mol_ids, save_path, file_name):
    with open(os.path.join(save_path, file_name + '.csv'), 'w') as f_out:
        for i in range(len(mol_ids)):
            f_out.write(str(mol_ids[i]) + '\n')




if __name__ == "__main__":
    # load cluster labels
    cluster_path = os.path.join(BASE_PATH, 'data/dataset_preparation/rna/rna_cluster_labels_2_6554_15.pkl')
    mol_ids = pickle_load(cluster_path)

    # load output path
    save_path = os.path.join(BASE_PATH + '/data/crossval/rna', 'all')
    os.makedirs(save_path, exist_ok=True)
    
    # save sets for folds
    for i in range(1, folds + 1):
        # generate test set (perform final model predictions to evaluate model)
        test_clusters_per_fold = fold_dict[i]
        test_mols_per_fold = list(itertools.chain(*[mol_ids[cluster] for cluster in test_clusters_per_fold]))
        print_csv(test_mols_per_fold, save_path, 'test_fold' + str(i))

        # generate validation set (assess training progress)
        leftover_keys = np.setdiff1d(validation_keys, i)
        # randomly select cluster for validation
        # attention: could potentially raise error if last leftover_keys in fold 7 is equal to test key --> re-run program
        validation_key = random.choice(leftover_keys)
        eval_clusters_per_fold = fold_dict[validation_key]
        eval_mols_per_fold = list(itertools.chain(*[mol_ids[cluster] for cluster in eval_clusters_per_fold]))
        print_csv(eval_mols_per_fold, save_path, 'eval' + str(validation_key) + '_fold' + str(i))
        # update validation key list to ensure each key only used once
        validation_keys.remove(validation_key)

        # generate training set (train model)
        train_keys = np.setdiff1d(list(fold_dict.keys()), [i, validation_key])
        train_clusters_per_fold = list(itertools.chain(*[fold_dict[tkey] for tkey in train_keys]))
        train_mols_per_fold = list(itertools.chain(*[mol_ids[cluster] for cluster in train_clusters_per_fold]))
        print_csv(train_mols_per_fold, save_path, 'train_fold' + str(i))
