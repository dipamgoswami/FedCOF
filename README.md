# FedCOF - Federated Learning with Covariances for Free [![Paper](https://img.shields.io/badge/arXiv-2412.14326-brightgreen)](https://arxiv.org/pdf/2412.14326)

## Code for NeurIPS 2025 paper - Covariances for Free: Exploiting Mean Distributions for Training-free Federated Learning

## Abstract
Using pre-trained models has been found to reduce the effect of data heterogeneity and speed up federated learning algorithms. Recent works have explored training-free methods using first- and second-order statistics to aggregate local client data distributions at the server and achieve high performance without any training. In this work, we propose a training-free method based on an unbiased estimator of class covariance matrices which only uses first-order statistics in the form of class means communicated by clients to the server. We show how these estimated class covariances can be used to initialize the global classifier, thus exploiting the covariances without actually sharing them. We also show that using only within-class covariances results in a better classifier initialization. Our approach improves performance in the range of 4-26% with exactly the same communication cost when compared to methods sharing only class means and achieves performance competitive or superior to methods sharing second-order statistics with dramatically less communication overhead. The proposed method is much more communication-efficient than federated prompt-tuning methods and still outperforms them. Finally, using our method to initialize classifiers and then performing federated fine-tuning or linear probing again yields better performance.


```
@inproceedings{goswami2025covariances,
  title={Covariances for Free: Exploiting Mean Distributions for Training-free Federated Learning},
  author={Dipam Goswami and Simone Magistri and Kai Wang and Bart{\l}omiej Twardowski and Andrew D. Bagdanov and Joost van de Weijer},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=71e1UmCMQ9}
}
```



## Training-Free Federated Learning using Pre-trained Models

To run the training-free methods (FedNCM, Fed3R and FedCOF), we provide the code to store the client features first based on the splits and then use different methods.

1. Generate the client splits for a given backbone, dataset and split specification as follows:

`python -u client_accumulate.py --dataset "cifar100" --model squeezenet --seed 0 --dataset_path ./data --alpha 0.1 --num_clients 100`

This script employs the **FedNCM strategy** to assign data to multiple clients using a Dirichlet distribution, controlled by the specified alpha and number of clients. The supported dataset are: ["inat", "cifar100", "imagenet-r", "cars", "cub"].

For `inat`, the number of clients is **fixed**, so the parameters `--alpha` and `--num_clients` do not affect them.

Script for generating the splits for all datasets are in the folder bash_for_datasets. 

The script generates two .pt files:

    train_client_data.pt: A dictionary in the format {"user": {"data": tensor, "labels": tensor}, ...}
    test_data.pt: A dictionary in the format {"data": tensor, "labels": tensor}

2. Use one of the training-free FL methods to obtain the classification performance. This script processes the feature dataset created by the client script and performs aggregation on the server.

`python -u server_aggregate.py -op ./output_folder --dataset "cifar100" --model "mobilenet" --seed 0 --seed_to_load 0 --approach "fedcof" --alpha_shrink 0.1 --src_path ./federated_features --alpha 0.1 --num_clients 100`

The script generate an experiment with a random name in the output folder. In this random experiments, there is a summary.csv with some necessary parameters used for running the experiments and the accuracy. src_path is the path where you store the federated_features via the script client_accumulate.py. The arguments - alpha, num_clients, seed_to_load are used to load the correct dataset. seed is the seed of the current experiment (usually is equal to seed_to_load).

Script for testing all the datasets are in the folder bash_for_testmethods. 

A Client class and a Server class for each training-free method could be found at federate_classes.py

3. Code for fine-tuning and linear-probing after FedCOF initialization will be uploaded soon.

