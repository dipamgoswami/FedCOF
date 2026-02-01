

for seed in 0 1 2 3 4; do
    for dataset in "inat"; do
        for model in "squeezenet" "mobilenet" "vit"; do
            python -u client_accumulate.py --dataset "$dataset" --model "$model" --seed "$seed" --dataset_path ... --num_workers 32
        done
    done
done