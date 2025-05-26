import os
import sys
import numpy as np
import torch
import argparse
from SC_FC_Exp import MyGNNModel_SC_FC
from others import clear_folder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup argument parser
parser = argparse.ArgumentParser(description='Connectomics Genomics Fuison for SZ Classfication')
parser.add_argument('--conn_modality', type=str, default='FC')
parser.add_argument('--fold', type=int, default=5)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--num_of_epoch', type=int, default=50)
parser.add_argument('--batchsize', type=int, default=128)
parser.add_argument('--drop_out_val', type=float, default=0.2)
parser.add_argument('--channel_size', type=int, default=32)
parser.add_argument('--class_number', type=int, default=2)
args = parser.parse_args()

for arg in vars(args):
    value = getattr(args, arg)
    print(f"{arg}: {value} (Type: {type(value).__name__})")

print(f'Modality: {args.conn_modality}')
print(f'Fold: {args.fold}')
print(f'Lr: {args.lr}')
print(f'Epoch: {args.num_of_epoch}')
print(f'Batch size: {args.batchsize}')
print(f'Channel size: {args.channel_size}')
print(f'Class number: {args.class_number}')
print(f'Drop out value: {args.drop_out_val}')
print(f'Device: {device}')

# # Setup experiment logging directory
base_path = "./Exp_Log/E{:03d}_log"
base_number = 0  # Initial log number
while os.path.exists(base_path.format(base_number)):
    base_number += 1
save_dir = base_path.format(base_number)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
print('model saved in:', save_dir)


GNN_dataset_path='./gnn_datasets'
Saved_model_path='./RGGC'
Saved_feat_embedd_path='./RGGC'
Loss_path='./RGGC'
Saved_exp_path='./RGGC'



############ For SC and FC  ############


model = MyGNNModel_SC_FC(modality=args.conn_modality, learning_rate=args.lr, 
                   dropout_value=args.drop_out_val, batch_size=args.batchsize, channel_size=args.channel_size, 
                   epoch_no=args.num_of_epoch, class_number=args.class_number, num_folds=args.fold, GNN_dataset_path=GNN_dataset_path, 
                   Saved_model_path=Saved_model_path,Saved_feat_embedd_path=Saved_feat_embedd_path,Loss_path=Loss_path,Saved_exp_path=Saved_exp_path)
model.SC_FC_exp_enh_model()