"""
Script serves three main purposes:
(i) Generate predictions with voxel-resolution for pre-computed voxelized grids
(ii) Visualize generated predictions
(iii) Analyze predictions: possible to compare predictions with voxel-resolution as well as predictions of reduced resolution to pre-computed ground truth with voxel-resolution

Author: Lukas Moeller
Date: 01/2022
"""


import argparse, os, logging, pickle, torch
import numpy as np
from moleculekit.util import writeVoxels
from moleculekit.tools.voxeldescriptors import getCenters
from torch.utils.data import DataLoader
from rna_binding.net import RNet1
from rna_binding.net_utils import PREDICTION_LOADER
from rna_binding.descriptor_generation.voxelize import channel_summed
from rna_binding.prediction.metrics import calculate_metrics, get_single_site_centers
from rna_binding.utils import(
    BASE_PATH,
    BOX_SIZE,
    DEVICE,
    VOXEL_SIZE,
    MODE_DICT,
    pickle_save,
    print_model_info,
)



def prediction_loop(model, loader, voxel_path, backup_path, save_vis=0, eval_res=1, resolution_decay=4, threshold=0.5, quantile=0.5, add_name=''):
    model = model.eval()
    eval_results = {}

    with torch.no_grad():
        # always use batch size == 1 when making predictions
        for mol, target, mol_name in loader:
            # send to gpu
            mol = mol.to(DEVICE)  
            # make prediction
            prediction = model(mol)
            # send to cpu for analysis
            mol, prediction = mol.cpu(), prediction.cpu()
            # convert from torch.tensor to np.array for analysis
            mol, target, prediction = np.array(mol).squeeze(), np.array(target).squeeze(), np.array(prediction).squeeze()
            # get single binding sites from target, prediction cubes
            tcoord, tlabel, tcenter, tnum = get_single_site_centers(target, threshold, quantile)
            pcoord, plabel, pcenter, pnum = get_single_site_centers(prediction, threshold, quantile)

            # analyze results
            if eval_res == 1:
                # remove orphan points from prediction
                orphan_label = -1
                index_pred = plabel == orphan_label
                if np.sum(index_pred) > 0:
                    tmp_udp = pcoord[index_pred].transpose()
                    prediction[tmp_udp[0], tmp_udp[1], tmp_udp[2]] = 0
                
                # calculate metrics
                results = calculate_metrics(target, prediction, tcoord, pcoord, tlabel, plabel, tcenter, pcenter, tnum, pnum, threshold)
                eval_results[str(mol_name[0])] = results
                # save backup of dict in case program crashes
                pickle_save(os.path.join(backup_path, 'eval_results' + str(add_name) + '_tmp.pkl'), eval_results)
            
            # save visualizations
            if save_vis == 1:
                # create directory to save visualizations
                out_path = os.path.join(voxel_path, str(mol_name[0]))
                os.makedirs(out_path, exist_ok=True)
                
                # get centers for each voxel in mol, target cube
                usercenters, _ = getCenters(boxsize=[BOX_SIZE, BOX_SIZE, BOX_SIZE], center=[0, 0, 0], voxelsize=VOXEL_SIZE)
                min_corner = usercenters.min(axis=0) - np.array([VOXEL_SIZE / 2] * 3)
                
                # get centers for each voxel in prediction, target cube
                usercenters_2, _ = getCenters(boxsize=[BOX_SIZE, BOX_SIZE, BOX_SIZE], center=[0, 0, 0], voxelsize=VOXEL_SIZE*resolution_decay)
                min_corner_2 = usercenters_2.min(axis=0) - np.array([VOXEL_SIZE*resolution_decay / 2] * 3)
                
                # summarize mol tensor for visualization
                tensor_mol_summed = channel_summed(mol)
                
                # write to cube files
                writeVoxels(tensor_mol_summed, os.path.join(out_path, 'mol.cube'), min_corner, [VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE])
                writeVoxels(target, os.path.join(out_path, 'target_all.cube'), min_corner_2, [VOXEL_SIZE*resolution_decay, VOXEL_SIZE*resolution_decay, VOXEL_SIZE*resolution_decay])
                writeVoxels(prediction, os.path.join(out_path, 'prediction_all.cube'), min_corner_2, [VOXEL_SIZE*resolution_decay, VOXEL_SIZE*resolution_decay, VOXEL_SIZE*resolution_decay])

                if tnum > 0:
                    for i in range(tnum):
                        index_target = tlabel == i
                        single_target = target.copy()
                        tmp_target = target.copy()
                        tmp_uct = tcoord[index_target].transpose()
                        tmp_target[tmp_uct[0], tmp_uct[1], tmp_uct[2]] = -10
                        single_target[tmp_target != -10] = 0
                        writeVoxels(single_target, os.path.join(out_path, 'target_' + str(i) + '.cube'), min_corner_2, [VOXEL_SIZE*resolution_decay, VOXEL_SIZE*resolution_decay, VOXEL_SIZE*resolution_decay])
                if pnum > 0:
                    for i in range(pnum):
                        index_pred = plabel == i
                        single_pred = prediction.copy()
                        tmp_pred = prediction.copy()
                        tmp_udp = pcoord[index_pred].transpose()
                        tmp_pred[tmp_udp[0], tmp_udp[1], tmp_udp[2]] = -10
                        single_pred[tmp_pred != -10] = 0
                        writeVoxels(single_pred, os.path.join(out_path, 'prediction_' + str(i) + '.cube'), min_corner_2, [VOXEL_SIZE*resolution_decay, VOXEL_SIZE*resolution_decay, VOXEL_SIZE*resolution_decay])
    
    return eval_results




if __name__ == "__main__":
    # bsub -n 4 -sp 90 -W 4:00 -R "rusage[mem=10240]" -R "rusage[ngpus_excl_p=1]" python prediction/predict.py -add_name _fold1 -model_path data/models/RNet1/rna_only/l5_r1_bs4_all/model_fold1_best.pkl -data_path data/crossval/rna/all/test_fold1.csv
    
    ###################################### user input ######################################
    # argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-nn', type=str, required=False, default='RNet1', help='Specify model architecture to be used. Make sure option matches model that is imported.')
    parser.add_argument('-nw', type=int, required=False, default=4, help='specify number of workers, make sure matches bsub -n')
    parser.add_argument("-mode", type=int, required=False, default=0, help='0: RNA, 1: proteins')
    parser.add_argument('-add_name', type=str, required=False, default='_fold1')
    parser.add_argument('-model_path', type=str, required=False, default='data/models/RNet1/finetuning/l5_bs4_all_C0207/model_fold5_best.pkl', help='')
    parser.add_argument('-data_path', type=str, required=False, default='data/crossval/rna/all/test.csv', help='')
    parser.add_argument("-save_vis", type=int, required=False, default=0)
    parser.add_argument("-a_res", type=int, required=False, default=1)
    parser.add_argument("-threshold", type=float, required=False, default=0.5)
    parser.add_argument("-quantile", type=float, required=False, default=0.5)
    # warning: do not use batch size other than 1 in current versions as metrics not optimized for batch operation
    parser.add_argument('-bs', type=int, required=False, default=1, help='specify batch size')
    args = parser.parse_args()
    nn_model = args.nn
    

    ######################################### main #########################################
    # silence moleculekit INFO logging/warnings and numexpr warnings
    logging.getLogger('moleculekit.molecule').setLevel(logging.ERROR)
    logging.getLogger('numexpr.utils').setLevel(logging.ERROR)

    # print path where utilized model is stored
    print(args.model_path)

    # generate output directories
    if 'finetuning' in args.model_path:
        vis_path = os.path.join(BASE_PATH, 'data/mol', str(MODE_DICT[args.mode]), 'prediction', 'finetuning')
    else:
        vis_path = os.path.join(BASE_PATH, 'data/mol', str(MODE_DICT[args.mode]), 'prediction', 'rna_only')
    res_path = os.path.dirname(os.path.join(BASE_PATH, args.model_path))
    os.makedirs(vis_path, exist_ok=True)

    # get path to data
    pred_path = os.path.join(BASE_PATH, args.data_path)

    # load data    
    prediction_data = PREDICTION_LOADER(pred_path, args.mode)
    prediction_loader = DataLoader(prediction_data, batch_size=args.bs, num_workers=args.nw, shuffle=False)

    # Load model
    nn_dict = {
        'RNet1': RNet1()
    }
    model = nn_dict[nn_model]
    model.load_state_dict(torch.load(os.path.join(BASE_PATH, args.model_path)))
    print_model_info(model)
    print('model is at: ', DEVICE)
    model = model.to(DEVICE)
    print('model is at: ', DEVICE)
    results = prediction_loop(model, prediction_loader, vis_path, res_path, save_vis=args.save_vis, eval_res=args.a_res, resolution_decay=4, threshold=args.threshold, quantile=args.quantile, add_name=str(args.add_name))

    # Save metric results
    pickle_save(os.path.join(res_path, 'eval_results' + str(args.add_name) + '.pkl'), results)
