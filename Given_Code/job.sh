#!/bin/bash
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=50g
#SBATCH -p qTRDGPUM
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -J MM_GNN
#SBATCH -e /data/users4/badhan/outdir/err%A.err
#SBATCH -o /data/users4/badhan/outdir/out%A.out
#SBATCH -A trends517s113 
#SBATCH --mail-user=bmazumder1@gsu.edu
#SBATCH --oversubscribe
sleep 10s
export OMP_NUM_THREADS=1
echo $HOSTNAME >&2
echo ARGS "${@:1}"
source /data/users4/badhan/anaconda3/bin/activate /data/users4/badhan/anaconda3/envs/badhan24
time python /data/users4/badhan/main_exp.py --conn_modality 'FC' --fold 10 --lr 1e-5 --num_of_epoch 500 --batchsize 32 --drop_out_val 0.2 --channel_size 32 --class_number 2
sleep 10s



