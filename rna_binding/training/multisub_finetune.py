""" Submit training jobs for all folds with one submission. Note that the script need to be adjusted based on the machine used. """

import subprocess, argparse

# bsub -n 4 -sp 90 -W 24:00 -R "rusage[mem=10240]" -R "rusage[ngpus_excl_p=1]" python training/finetune.py -nn RNet1 -bs 4 -nw 4 -start_epochs 0 -end_epochs 3001 -save_int 10 -lr 1e-5 -opt_mode all -fold 1 -add_name _0117 -model_path data/models/RNet1/rna_only/l5_r1_bs4_all/tmp_log/model_fold1_ep2000.pkl

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-start_epochs', type=int, required=False, default=0, help='starting number of epochs (for re-submission of jobs)')
    parser.add_argument('-end_epochs', type=int, required=False, default=2001, help='maximal number of epochs')
    parser.add_argument('-gamma_f', type=float, required=False, default=0.4, help='gamma factor for lr scheduler')
    parser.add_argument('-step_s', type=int, required=False, default=2000, help='step size for lr scheduler')
    parser.add_argument('-add_name', type=str, required=False, default='', help='name to be appended when saving models')
    parser.add_argument('-model_path', type=str, required=False, default='', help='path to pretrained model. if empty, no pre-trained model will be loaded')
    args = parser.parse_args()
    
    # submission of multiple processes
    for i in range(1, 8):
        if 'tmp_log' in str(args.model_path):
            path2model = str(args.model_path) + 'model_fold' + str(i) + '_ep' + str(args.start_epochs) + '.pkl'
        else:
            path2model = ''
        
        cmd_str = [
            "bsub",
            "-n",
            "4",
            "-W",
            "24:00",
            "-sp",
            "90",
            "-R",
            "rusage[mem=10240]",
            "-R",
            "rusage[ngpus_excl_p=1]",
            "python",
            "training/finetune.py",
            "-start_epochs",
            str(args.start_epochs),
            "-end_epochs",
            str(args.end_epochs),
            "-gamma_f",
            str(args.gamma_f),
            "-step_s",
            str(args.step_s),
            "-fold",
            str(i),
            "-add_name",
            str(args.add_name),
            "-model_path",
            str(path2model),
        ]
        completed_process = subprocess.run(cmd_str)