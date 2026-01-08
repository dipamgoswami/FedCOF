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

Code for all training-free FL methods will be available soon.
