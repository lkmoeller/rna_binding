import subprocess, argparse
import numpy as np

# bsub -n 4 -sp 90 -W 4:00 -R "rusage[mem=10240]" -R "rusage[ngpus_excl_p=1]" python prediction/predict.py -add_name _fold1 -model_path data/models/RNet1/rna_only/l5_r1_bs4_all -data_path data/crossval/rna/all/test_fold1.csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-save_vis', type=int, required=False, default=0, help='')
    parser.add_argument('-a_res', type=int, required=False, default=1, help='')
    parser.add_argument('-threshold_min', type=float, required=False, default=0.5, help='')
    parser.add_argument('-threshold_max', type=float, required=False, default=0.5, help='')
    parser.add_argument('-threshold_step', type=float, required=False, default=0.2, help='')
    parser.add_argument('-quantile_min', type=float, required=False, default=0.5, help='')
    parser.add_argument('-quantile_max', type=float, required=False, default=0.5, help='')
    parser.add_argument('-quantile_step', type=float, required=False, default=0.5, help='')
    parser.add_argument('-add_name', type=str, required=False, default='', help='name to be appended when saving models')
    parser.add_argument('-model_path', type=str, required=False, default='', help='path to pretrained model. if empty, no pre-trained model will be loaded')
    args = parser.parse_args()
    
    # submission of multiple processes
    for i in range(1, 8):
        path2model = str(args.model_path) + '/model_fold' + str(i) + '_best.pkl'
        path2data = 'data/crossval/rna/all/test_fold' + str(i) + '.csv'

        for t in np.arange(float(args.threshold_min), float(args.threshold_max) + float(args.threshold_step), float(args.threshold_step)):
            for q in np.arange(float(args.quantile_min), float(args.quantile_max) + float(args.quantile_step), float(args.quantile_step)):
                add2name = '_fold' + str(i) + '_t' + str(int(t*10)) + '_q' + str(int(q*10))

                cmd_str = [
                    "bsub",
                    "-n",
                    "4",
                    "-W",
                    "4:00",
                    "-sp",
                    "90",
                    "-R",
                    "rusage[mem=10240]",
                    "-R",
                    "rusage[ngpus_excl_p=1]",
                    "python",
                    "prediction/predict.py",
                    "-save_vis",
                    str(args.save_vis),
                    "-a_res",
                    str(args.a_res),
                    "-threshold",
                    str(t),
                    "-quantile",
                    str(q),
                    "-data_path",
                    str(path2data),
                    "-add_name",
                    str(add2name),
                    "-model_path",
                    str(path2model),
                ]
                completed_process = subprocess.run(cmd_str)