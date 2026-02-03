#!/bin/bash

for approach in "fedncm" "fedmd" "fed3r"; do
    for seed in 0 1 2 3 4; do
        
        for dataset in "cifar100" "imagenet-r" "cars" "cub"; do
    
                python -u server_aggregate.py --dataset "$dataset" --model "vit" --seed "$seed" --seed_to_load "$seed" --approach $approach --gamma_shrink 1.0
            
        done
    done
done 
