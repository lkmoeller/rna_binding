"""
Definition of RNet architecture

Authors: Lukas Moeller, Lorenzo Guerci
Date: 01/2022
"""


import torch
import torch.nn as nn
import torch.nn.functional as F



class RNet1(nn.Module):
    """
    Resolution decrease by factor 4
    model parameters:  277,297
    """
    
    def __init__(self, in_channels=8, out_channels=1):
        super(RNet1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_1 = nn.Conv3d(self.in_channels, 16, kernel_size=5, stride=1, padding=2)
        self.conv_2 = nn.Conv3d(16, 32, kernel_size=5, stride=1, padding=2)
        self.conv_3 = nn.Conv3d(32, 32, kernel_size=5, stride=1, padding=2)
        self.pool_1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1, dilation=1)
        self.conv_4 = nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv_5 = nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv_6 = nn.Conv3d(32, 16, kernel_size=3, stride=1, padding=1)
        self.pool_2 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1, dilation=1)
        self.conv_7 = nn.Conv3d(16, self.out_channels, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv_1(x)) 
        x = F.relu(self.conv_2(x))
        x = F.relu(self.conv_3(x))
        x = self.pool_1(x)
        x = F.relu(self.conv_4(x))
        x = F.relu(self.conv_5(x))
        x = F.relu(self.conv_6(x))
        x = self.pool_2(x)
        x = torch.sigmoid(self.conv_7(x)) 
        return x.squeeze(1)



if __name__ == "__main__":
    import argparse, os
    from torch.utils.data import DataLoader
    from rna_binding.net_utils import  LOADER
    from rna_binding.utils import(
        BASE_PATH,
        print_model_info
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("-bs", type=int, required=False, default=2, help='batch size')
    parser.add_argument("-nw", type=int, required=False, default=1, help='number of workers')   
    parser.add_argument("-mode", type=int, required=False, default=0, help='running mode: 0 = RNA, 1 = Protein')
    parser.add_argument("-path", type=str, required=False, default='')
    parser.add_argument("-use_rot", type=int, required=False, default=0)
    args = parser.parse_args()

    train_path = os.path.join(BASE_PATH, args.path)
    train_data = LOADER(train_path, args.mode, args.use_rot)
    train_loader = DataLoader(train_data, batch_size=args.bs, num_workers=args.nw, shuffle=True)
    print("Number of training data ", len(train_loader))

    # Load model
    model = RNet1()
    print_model_info(model)

    # loop over batches
    for mol, target in train_loader:
        print(mol.size())
        prediction = model(mol)
        print(prediction.size(), target.size())
