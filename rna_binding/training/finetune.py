"""
Train 3D-CNN on RNA structures to predict ligand binding sites

Authors: Lukas Moeller
Date: 01/2022
"""



import argparse, os, logging, pickle, math, torch, glob
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from rna_binding.net import RNet1
from rna_binding.net_utils import LOADER
from rna_binding.utils import(
    BASE_PATH,
    DEVICE,
    pickle_load,
    print_model_info
)



def train_loop(model, loader, optimizer, criterion):
    model = model.train()
    training_loss = []

    for mol, target in loader:
        optimizer.zero_grad() # only in train_loop

        mol = mol.to(DEVICE)  
        target = target.to(DEVICE)
        prediction = model(mol)

        loss = criterion(prediction, target)
        loss.backward() # only in train_loop
        optimizer.step() # only in train_loop
        training_loss.append(loss.item())
        
    return np.mean(training_loss)


def eval_loop(model, loader, criterion):
    model = model.eval()
    eval_loss = []

    with torch.no_grad():
        for mol, target in loader:
            mol = mol.to(DEVICE)  
            target = target.to(DEVICE)
            prediction = model(mol)

            loss = criterion(prediction, target)
            eval_loss.append(loss.item())

    return np.mean(eval_loss)


if __name__ == "__main__":

    # bsub -n 4 -sp 90 -W 24:00 -R "rusage[mem=10240]" -R "rusage[ngpus_excl_p=1]" python training/finetune.py -nn RNet1 -bs 4 -nw 4 -start_epochs 0 -end_epochs 3001 -save_int 10 -lr 1e-5 -opt_mode all -fold 1 -add_name _0117 -model_path data/models/RNet1/rna_only/l5_r1_bs4_all/tmp_log/model_fold1_ep2000.pkl

    # silence moleculekit INFO logging
    logging.getLogger('moleculekit.molecule').setLevel(logging.ERROR)

    # argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-nn', type=str, required=False, default='RNet1', help='Specify model architecture to be used. Make sure option matches model that is imported.')
    parser.add_argument('-bs', type=int, required=False, default=4, help='specify batch size')
    parser.add_argument('-nw', type=int, required=False, default=4, help='specify number of workers, make sure matches bsub -n')
    parser.add_argument('-start_epochs', type=int, required=False, default=0, help='starting number of epochs (for re-submission of jobs)')
    parser.add_argument('-end_epochs', type=int, required=False, default=2001, help='maximal number of epochs')
    parser.add_argument('-save_int', type=int, required=False, default=10, help='interval to save data')
    parser.add_argument('-lr', type=float, required=False, default=1e-5, help='learning rate')
    parser.add_argument('-gamma_f', type=float, required=False, default=0.4, help='gamma factor for lr scheduler')
    parser.add_argument('-step_s', type=int, required=False, default=2000, help='step size for lr scheduler')
    parser.add_argument('-opt_mode', type=str, required=False, default='all', help='if opt_mode == "opt_n": only n proteins used to speed up training process. Use this mode for model architecture optimization.')
    parser.add_argument('-fold', type=int, required=False, default=1, help='fold to be used in crossvalidation scheme')
    parser.add_argument('-add_name', type=str, required=False, default='', help='name to be appended when saving models')
    parser.add_argument('-model_path', type=str, required=False, default='', help='path to pretrained model. if empty, no pre-trained model will be loaded')
    
    args = parser.parse_args()
    nn_model = args.nn

    # load paths to training and validation data
    train_path = os.path.join(BASE_PATH, 'data/crossval/rna', str(args.opt_mode), 'train_fold'+str(args.fold)+'.csv')
    eval_path = glob.glob(os.path.join(BASE_PATH, 'data/crossval/rna', str(args.opt_mode), 'eval*_fold'+str(args.fold)+'.csv'))[0]
    
    # generate paths/directories to save training stats, trained models
    if len(str(args.model_path)) == 0 or 'rna_only' in str(args.model_path):
        log_path = os.path.join(BASE_PATH, 'data/models', nn_model, 'rna_only', 'l' + str(int(round(-math.log(args.lr, 10)))) + '_bs' + str(args.bs) + '_' + str(args.opt_mode) + str(args.add_name))
    else:
        log_path = os.path.join(BASE_PATH, 'data/models', nn_model, 'finetuning', 'l' + str(int(round(-math.log(args.lr, 10)))) + '_bs' + str(args.bs) + '_' + str(args.opt_mode) + str(args.add_name))
    tmp_log_path = os.path.join(log_path, 'tmp_log')
    
    # check if directories already exist, generate directories
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(tmp_log_path, exist_ok=True)

    # load data
    loader_mode = 0 # use RNA
    train_data = LOADER(train_path, loader_mode)
    train_loader = DataLoader(train_data, batch_size=args.bs, num_workers=args.nw, shuffle=True)
    eval_data = LOADER(eval_path, loader_mode)
    eval_loader = DataLoader(train_data, batch_size=args.bs, num_workers=args.nw, shuffle=True)

    # load model
    nn_dict = {
        'RNet1': RNet1()
    }
    model = nn_dict[nn_model]
    if len(str(args.model_path)) > 0:
        model.load_state_dict(torch.load(os.path.join(BASE_PATH, args.model_path)))
    print_model_info(model)
    print('model is at: ', DEVICE)
    model = model.to(DEVICE)
    print('model is at: ', DEVICE)

    # set up optimizer, loss function
    if int(args.start_epochs) > 1:
        a_lr = args.lr * (args.gamma_f ** np.floor(int(args.start_epochs)/args.step_s))
        optimizer = torch.optim.Adam(model.parameters(), lr=a_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-6)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-6)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_s, gamma=args.gamma_f, verbose=True)

    if int(args.start_epochs) > 1:
        train_losses, eval_losses = pickle_load(os.path.join(tmp_log_path, 'loss_fold' + str(args.fold) + '_ep' + str(args.start_epochs) + '.pkl'))
        min_loss = np.min(eval_losses)
    else:
        train_losses, eval_losses = [], []
        min_loss = 10000000

    for epoch in range(args.start_epochs + 1, args.end_epochs):
        print(f"{nn_model}, {args.lr}: Epoch {epoch}/{args.end_epochs} --> {log_path}")

        # train on RNA data
        train_loss = train_loop(model, train_loader, optimizer, criterion)
        train_losses.append(train_loss)

        # eval on RNA data
        eval_loss = eval_loop(model, eval_loader, criterion)
        eval_losses.append(eval_loss)

        # adjust learning rate
        scheduler.step()

        # save if min eval loss
        if eval_loss < min_loss:
            # set new min eval loss
            min_loss = eval_loss

            # save model
            torch.save(model.state_dict(), os.path.join(log_path, 'model_fold' + str(args.fold) + '_best.pkl'),)

            # save losses
            with open(os.path.join(log_path, 'loss_fold' + str(args.fold) + '_best.pkl'), "wb") as handle:
                pickle.dump((train_losses, eval_losses), handle)
        
        # save model and stats every n epochs
        if epoch % args.save_int == 0:
            # save model
            torch.save(model.state_dict(), os.path.join(tmp_log_path, 'model_fold' + str(args.fold) + '_ep' + str(epoch) + '.pkl'),)

            # save losses
            with open(os.path.join(tmp_log_path, 'loss_fold' + str(args.fold) + '_ep' + str(epoch) + '.pkl'), "wb") as handle:
                pickle.dump((train_losses, eval_losses), handle)
