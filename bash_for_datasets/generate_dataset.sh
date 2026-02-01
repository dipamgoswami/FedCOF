

for seed in 0 1 2 3 4; do
    for dataset in "cifar100" "imagenet-r" "cars" "cub"; do
        for model in "squeezenet" "mobilenet" "vit"; do
            python -u client_accumulate.py --dataset "$dataset" --model "$model" --seed "$seed"
        done
    done
done