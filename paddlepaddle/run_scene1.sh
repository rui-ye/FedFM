client=10
sample_fraction=1.0
START_TIME=`date +%s`
cg_tau=0.1 
lam_fm=50.0 
start_ep_fm=20
use_project_head=0 
fm_avg_anchor=0 
dataset='cifar10' 
device='gpu:0' 
model='resnet18_7 '
alg='fedfm' 
lr=0.01 
epochs=10 
comm_round=100 
n_parties=10 
partition='noniid'
beta=0.5 
dir_path=./logs/${dataset}_${partition}_${model}_beta${beta}_it${epochs}_c${client}_p${sample_fraction}
# datadir='/GPFS/data/ruiye/fssl/dataset'


nohup python -u main_new.py --start_ep_fm $start_ep_fm --comm_round $comm_round --dataset $dataset --device $device --partition $partition --model $model --n_parties $client --sample_fraction $sample_fraction --epochs $epochs --beta $beta > $dir_path/fedavg_${START_TIME}.log 
