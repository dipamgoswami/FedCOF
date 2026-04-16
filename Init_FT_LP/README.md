
# FEDERATED LEARNING WITH COVARIANCES FOR FREE (FEDCOF)

This repository contains the source code for FedCOF.
Our implementation modifies the FedNCM code which uses the [FL Sim](https://github.com/facebookresearch/FLSim) source code. 

### Environment
Setup a conda environment with the specified requirements

`pip install -r requirements.txt
`

### How to Set up a Run:
FL Sim sets up run configurations using `config.py`, additionally we have implemented the option to configure run settings
using the command line. To see `config.py` <=> command line equivalences, see method `set_cfg_from_cl()` in `utils.py`.

If you do not supply a command line argument, configuration will defer to the value set in `config.py`.

#### Sample Run Command for FedCOF:
`CUDA_VISIBLE_DEVICES=0 python federated_main_round.py --wandb=False --epochs=0 --num_clients=100 --clients_per_round=30 --dataset=cifar100 --pretrained=1 --algorithm=lp --fl_algorithm=fedavg --optimizer=sgd --alpha=0.1 --fedcof=1 --model=squeezenet --subsets=1 --seed=1
`

#### Sample Run Command for FedNCM:
`CUDA_VISIBLE_DEVICES=0 python federated_main_round.py --wandb=False --epochs=0 --num_clients=100 --clients_per_round=30 --dataset=cifar100 --pretrained=1 --ncm=1 --algorithm=lp --fl_algorithm=fedavg --optimizer=sgd --alpha=0.1 --model=squeezenet --subsets=1 --seed=1
`

#### Sample Run Command for Fed3R:
`CUDA_VISIBLE_DEVICES=0 python federated_main_round.py --wandb=False --epochs=0 --num_clients=100 --clients_per_round=30 --dataset=cifar100 --pretrained=1 --algorithm=lp --fl_algorithm=fedavg --optimizer=sgd --alpha=0.1 --model=squeezenet --subsets=1 --fed3r=1 --seed=1
`

#### Sample Run Command for FedCOF+FT (FedAvg):
`CUDA_VISIBLE_DEVICES=0 python federated_main_round.py --wandb=False --epochs=200 --num_clients=100 --clients_per_round=30 --dataset=cifar100 --local_ep=1 --pretrained=1 --algorithm=ft --fl_algorithm=fedavg --optimizer=sgd --alpha=0.1 --client_lr=0.001 --fedcof=1 --model=squeezenet --subsets=1 --seed=1
`

#### Sample Run Command for FedCOF+FT (FedAdam):
`CUDA_VISIBLE_DEVICES=0 python federated_main_round.py --wandb=False --epochs=200 --num_clients=100 --clients_per_round=30 --dataset=imagenet-r --local_ep=1 --pretrained=1 --algorithm=ft --fl_algorithm=fedadam --optimizer=sgd --alpha=0.1 --client_lr=0.0001 --server_lr=0.0001 --fedcof=1 --model=squeezenet --subsets=1 --seed=1
`

#### Sample Run Command for FedCOF+LP (FedAvg):
`CUDA_VISIBLE_DEVICES=0 python federated_main_round.py --wandb=False --epochs=200 --num_clients=100 --clients_per_round=30 --dataset=cifar100 --local_ep=1 --pretrained=1 --algorithm=lp --fl_algorithm=fedavg --optimizer=sgd --alpha=0.1 --client_lr=0.001 --fedcof=1 --model=squeezenet --subsets=1 --seed=1
`

#### wandb:
This code base works with wandb logging, to enable it, set the appropriate command line options, or the configs in the 
wandb section of `config.py`.

