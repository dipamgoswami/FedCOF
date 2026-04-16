#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import flsim.configs  # noqa
import torch
import timm
import random
import numpy as np
import pickle
from torch.utils.data import DataLoader,Dataset
from flsim.data.data_sharder import SequentialSharder, RandomSharder
from flsim.channels.base_channel import FLChannelConfig
from flsim.clients.base_client import ClientConfig
from flsim.servers.sync_servers import SyncServerConfig
from flsim.active_user_selectors.simple_user_selector import (
    SequentialActiveUserSelectorConfig, UniformlyRandomActiveUserSelectorConfig  
)
from flsim.optimizers.local_optimizers import LocalOptimizerSGDConfig, LocalOptimizerAdamConfig
from flsim.interfaces.metrics_reporter import Channel
from flsim.utils.config_utils import fl_config_from_json
from omegaconf import DictConfig, OmegaConf
from config import json_config
import wandb
from torchvision.models import squeezenet1_1, resnet18, ResNet18_Weights, resnet50, ResNet50_Weights, MobileNet_V2_Weights
from torchvision.models.squeezenet import SqueezeNet1_1_Weights
from torchvision.models import mobilenet_v2
from sync_trainer_round import SyncTrainer, SyncTrainerConfig
import os
from utils import validata_dataset_params, build_data_provider, FLModel, MetricsReporter, wandb_setup, set_cfg_from_cl, \
    inference, shrink_cov
import torch.nn.functional as F
from flsim.optimizers.server_optimizers import FedAvgWithLROptimizerConfig, FedAdamOptimizerConfig, FedAvgOptimizerConfig
from torch.distributions.multivariate_normal import MultivariateNormal


def main(cfg,
    use_cuda_if_available: bool = True,
) -> None:
    cuda_enabled = torch.cuda.is_available() and use_cuda_if_available
    device = torch.device(f"cuda:{0}" if cuda_enabled else "cpu")

    print('Seed: ',cfg.dataset.seed,' Dataset: ',cfg.dataset.name,' Model: ',cfg.trainer.model)
    if cuda_enabled:
        torch.cuda.manual_seed(cfg.dataset.seed)
        torch.manual_seed(cfg.dataset.seed)
        torch.cuda.manual_seed_all(cfg.dataset.seed)
    else:
        torch.manual_seed(cfg.dataset.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(cfg.dataset.seed)
    random.seed(cfg.dataset.seed)

    if cfg.trainer.pretrained:
        if cfg.wandb.offline:
            if cfg.trainer.model == 'resnet18':
                model = resnet18()
                model.load_state_dict(torch.load("./models/pretrained_resnet.pt"))
            else:
                model = squeezenet1_1()
                model.load_state_dict(torch.load("./models/pretrained_squeeze.pt"))
        else:
            if cfg.trainer.model == 'resnet18':
                model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            elif cfg.trainer.model == 'resnet50':
                model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            elif cfg.trainer.model == 'vit':
                model = timm.create_model("vit_base_patch16_224_in21k",pretrained=True).cuda()
            elif cfg.trainer.model == 'mobilenet':
                model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
            else:
                model = squeezenet1_1(weights=SqueezeNet1_1_Weights.DEFAULT)
    else:
        if 'resnet18' in cfg.trainer.model:
            model = resnet18()
        elif 'resnet50' in cfg.trainer.model:
            model = resnet50()
        else:
            model = squeezenet1_1()

    # Algorithms: ft--> fine tune all, lp--> only train last layer
    if 'lp' in cfg.trainer.algorithm:
        for name, param in model.named_parameters():
            param.requires_grad = False


    # replace classifier with randomly initialized classifier of appropriate size
    if 'resnet' in cfg.trainer.model:
        model.fc = torch.nn.Linear(512, cfg.dataset.num_classes, bias=False)
        feat_dims = 512
    elif 'vit' in cfg.trainer.model:
        model.head = torch.nn.Linear(768, cfg.dataset.num_classes, bias=False)
        feat_dims = 768
    elif 'mobilenet' in cfg.trainer.model:
        feat_dims = 1280
        model.classifier = torch.nn.Sequential(
            # nn.Dropout(p=dropout),
            torch.nn.Identity(),
            torch.nn.Identity(),
            torch.nn.Linear(1280, cfg.dataset.num_classes, bias=False),
        )
        cfg.trainer.shrink = 0.1
    else:
        model.classifier = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(512, cfg.dataset.num_classes, bias=False)
        )
        feat_dims = 512

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('trainable parameters: ',params)

    model.eval()
    # wandb setup
    cfg.wandb.activate = False
    
    if cfg.wandb.activate:
        run_dir = f'scratch/{os.environ.get("USER", "username")}/{cfg.wandb.run_name}'
        if not os.path.isdir(run_dir):
            os.makedirs(run_dir, mode=0o755, exist_ok=True)
        wandb_setup(cfg)
        wandb.watch(model, log_freq=cfg.wandb.log_freq)
        
    if cfg.trainer.epochs > 0:
        eval_epoch = 1
    else:
        eval_epoch = 0

    global_model = FLModel(model, device)
    if cuda_enabled:
        global_model.fl_cuda()

    fl_data_loader, data_provider = build_data_provider(
        local_batch_size=cfg.trainer.client.local_bs,
        num_clients=cfg.trainer.total_users,
        dataset=cfg.dataset.name,
        num_classes=cfg.dataset.num_classes,
        alpha=cfg.dataset.alpha,
        num_client_samples=cfg.dataset.num_client_samples,
        drop_last=False,
    )
    length = len(fl_data_loader.train_dataset)
    
    testloader = torch.utils.data.DataLoader(
        fl_data_loader.test_dataset, batch_size=128, shuffle=False, num_workers=2)

    
    # set optimizer
    # TODO: figure out how to set the optimizers using definition in the config
    if cfg.set_optimizer == 'sgd':
        opt = LocalOptimizerSGDConfig(lr=cfg.trainer.client.optimizer.lr)
    elif cfg.set_optimizer == 'adam':
        # note: ADAM automatically sets weight decay 1e-5
        opt = LocalOptimizerAdamConfig(lr=cfg.trainer.client.optimizer.lr)
    else:
        raise Exception(f"value {cfg.set_optimizer._base_} specified for cfg.set_optimizer._base_ is invalid")

    # set server optimizer based on fl_algorithm
    if cfg.trainer.fl_algorithm.lower() == 'fedavgm':
        server_opt = FedAvgWithLROptimizerConfig(lr=cfg.trainer.server.server_optimizer.lr,
                                                 momentum=cfg.trainer.server.server_optimizer.momentum)
    elif cfg.trainer.fl_algorithm.lower() == 'fedadam':
        server_opt = FedAdamOptimizerConfig(lr=cfg.trainer.server.server_optimizer.lr,
                   weight_decay=cfg.trainer.server.server_optimizer.weight_decay,
                   beta1=cfg.trainer.server.server_optimizer.beta1,
                   beta2=cfg.trainer.server.server_optimizer.beta2,
                   eps=cfg.trainer.server.server_optimizer.eps,)
    else:
        server_opt = FedAvgOptimizerConfig()
    
    # this version of trainer used in all 4 algorithm cases
    trainer = SyncTrainer(
            model=global_model,
            cuda_enabled=False,
            **OmegaConf.structured(SyncTrainerConfig(
                epochs=cfg.trainer.epochs,
                do_eval=cfg.trainer.do_eval,
                always_keep_trained_model=False,
                train_metrics_reported_per_epoch=cfg.trainer.train_metrics_reported_per_epoch,
                eval_epoch_frequency=eval_epoch,
                report_train_metrics=cfg.trainer.report_train_metrics,
                report_train_metrics_after_aggregation=cfg.trainer.report_train_metrics_after_aggregation,
                client=ClientConfig(
                    epochs=cfg.trainer.client.epochs,
                    optimizer=opt,
                    lr_scheduler=cfg.trainer.client.lr_scheduler,
                    shuffle_batch_order=False,
                ),
                channel=FLChannelConfig(),
                server=SyncServerConfig(
                    active_user_selector=UniformlyRandomActiveUserSelectorConfig(),
                    server_optimizer=server_opt
                ),
                users_per_round=cfg.trainer.users_per_round,
                dropout_rate=cfg.trainer.dropout_rate,
                wandb=cfg.wandb,
                wsm=cfg.trainer.wsm,
            )),
        )
    
    metrics_reporter = MetricsReporter([Channel.TENSORBOARD, Channel.STDOUT], wandb_dict=cfg.wandb)  
    
    # Initialization in multiple rounds
    trainer.sampled_clients = []
    total_class_sums = {new_list: torch.zeros(feat_dims).to(device) for new_list in range(cfg.dataset.num_classes)}
    total_count_cln = {new_list: [] for new_list in range(cfg.dataset.num_classes)}
    total_mu = {new_list: [] for new_list in range(cfg.dataset.num_classes)}
    est_covs = {new_list: torch.eye(feat_dims).to(device) for new_list in range(cfg.dataset.num_classes)}
    est_covs_no = {new_list: torch.eye(feat_dims).to(device) for new_list in range(cfg.dataset.num_classes)}
    mu_avg = {new_list: torch.zeros(feat_dims).to(device) for new_list in range(cfg.dataset.num_classes)}
    total_counts = {new_list: 0 for new_list in range(cfg.dataset.num_classes)}
    gram_mat_rr = 0
    
    if not cfg.trainer.fedcof and not cfg.trainer.ncm_init and not cfg.trainer.fed3r:
        total_epochs = 0
    else:
        total_epochs = 100
    for init_epoch in range(total_epochs):  # multi-step initialization
        already_sampled = len(trainer.sampled_clients)
        if len(trainer.sampled_clients) == data_provider.num_train_users():
            break
        mu, cov_mats, count = trainer.cp_init(cfg.dataset.num_classes, device, global_model.model, cfg.trainer.users_per_round, 
                                            cfg.trainer.model, cfg.trainer.need_cov, cfg.trainer.fed3r, cfg.trainer.subsets, feat_dims, 
                                            data_provider=data_provider)

        if already_sampled == len(trainer.sampled_clients):
            continue

        cc = []
        num_subsets = cfg.trainer.subsets

        for idx in range(cfg.dataset.num_classes):
            # print(idx)
            feats, mu_sum_cln, sum_cln = [], [], []
            all_mu, count_cln = [], []
            est_covs_, t2 = [], []
            total_count = 0
            for cln in range(cfg.trainer.users_per_round*num_subsets):

                if len(mu[cln][idx]) > 0:
                    all_mu.append(mu[cln][idx].clone().detach())
                    count_cln.append(count[cln][idx])
                    mu_sum_cln.append(count[cln][idx]*mu[cln][idx])

            if len(all_mu) > 0:
                if len(total_count_cln[idx]) == 0:
                    total_count_cln[idx] = torch.tensor(count_cln)
                else:
                    total_count_cln[idx] = torch.cat((total_count_cln[idx], torch.tensor(count_cln)))
                total_counts[idx] = total_count_cln[idx].sum()

                if len(total_mu[idx]) == 0:
                    total_mu[idx] = torch.stack(all_mu)
                else:
                    total_mu[idx] = torch.cat((total_mu[idx], torch.stack(all_mu)))
                
                total_class_sums[idx] += torch.stack(mu_sum_cln).sum(0)

                mu_avg[idx] = total_class_sums[idx]/total_counts[idx]

            cc.append(len(total_mu[idx]))

            for cln in range(cfg.trainer.users_per_round*num_subsets):
                if len(mu[cln][idx]) > 0:
                    
                    if cfg.trainer.need_cov and not cfg.trainer.fed3r: 
                        t2.append((count[cln][idx]/(total_count-1))*torch.matmul(mu[cln][idx].unsqueeze(1),mu[cln][idx].unsqueeze(0)))
                        est_covs_.append(((count[cln][idx]-1)/(total_count-1))*cov_mats[cln][idx])

            # aggregating class covariances from clients similar to CCVR for the oracle setting
            if cfg.trainer.need_cov and not cfg.trainer.fed3r:  
                t3 = (total_count/(total_count-1))*torch.matmul(mu_avg[idx].unsqueeze(1),mu_avg[idx].unsqueeze(0))
                cov_x = torch.stack(est_covs_).sum(0) + torch.stack(t2).sum(0) - t3
                est_covs[idx] = cov_x

            if not cfg.trainer.need_cov and not cfg.trainer.fed3r:
                if len(all_mu) > 0:
                    est_mu = mu_avg[idx]  #all_mu.mean(0)
                    if len(total_mu[idx]) == 1:
                        all_mu_aug = []
                        # print('Assuming euclidean distribution')
                        for mu1 in all_mu:
                            all_mu_aug.append(mu1)
                            for tmp in range(5):
                                all_mu_aug.append(mu1+(0.01**0.5)*torch.randn(mu1.shape).to(device))

                        all_mu_aug = torch.stack(all_mu_aug)
                        est_covs_no[idx] = shrink_cov(torch.cov(all_mu_aug.T), device, cfg.trainer.shrink)
                    else:
                        mu_x = total_mu[idx] - est_mu
                        mu_x *= torch.sqrt(total_count_cln[idx])[:, None].to(device)
                        inside = torch.mm(mu_x.T, mu_x) 

                        delta = inside/(len(total_mu[idx])-1)
                        est_covs_no[idx] = shrink_cov(delta, device, cfg.trainer.shrink) 
            
        if cfg.trainer.need_cov:
            est_covs_cls = est_covs
        else:
            est_covs_cls = est_covs_no

        if not cfg.trainer.fed3r:
            g_means, all_cls_mu = [], []
            total_feats = 0
            for idx in range(cfg.dataset.num_classes):
                all_cls_mu.append(mu_avg[idx])
                g_means.append(mu_avg[idx]*total_counts[idx])
                total_feats += total_counts[idx]
            g_means = torch.stack(g_means).sum(0)/total_feats

            # all_cls_mu = torch.stack(all_cls_mu)
            # mu_cls = all_cls_mu - g_means
            # mu_cls *= torch.sqrt(total_counts)[:, None].to(device)
            inside = total_feats*torch.mm(g_means.unsqueeze(1), g_means.unsqueeze(0)) # + torch.mm(mu_cls.T, mu_cls)

            est_gram = torch.zeros(feat_dims, feat_dims).to(device)
            for idx in range(cfg.dataset.num_classes):
                est_gram += (total_counts[idx]-1)*est_covs_cls[idx]
            est_gram = est_gram + inside

        print("Total means shared - ",torch.tensor(cc).sum().item())
            
        if cfg.trainer.ncm_init:
            mu, all_means =  [], []
            for v in mu_avg.values():
                mu.append(v)
            mu = torch.stack(mu)

            if 'resnet' in cfg.trainer.model:
                global_model.model.fc.weight.data = torch.nn.functional.normalize(mu)
            elif 'vit' in cfg.trainer.model:
                global_model.model.head.weight.data = torch.nn.functional.normalize(mu)
            else:
                global_model.model.classifier[2].weight.data = torch.nn.functional.normalize(mu)
        
            test_accuracy, test_loss = inference(global_model.model, testloader, device)
            print('FedNCM test acc on inference: ', test_accuracy)
            if cfg.wandb.activate and cfg.trainer.pretrained:
                wandb.log({'init acc': test_accuracy*100})

        if cfg.trainer.fedcof:
            class_means = []
            gram_mat = shrink_cov(est_gram, device, cfg.trainer.shrink)
            
            inv_gram_mat = torch.linalg.pinv(gram_mat).float().to(device)
            for idx in range(cfg.dataset.num_classes):
                class_means.append(torch.matmul(inv_gram_mat, total_class_sums[idx]))
            class_means = torch.stack(class_means)

            if 'resnet' in cfg.trainer.model:
                global_model.model.fc.weight.data = torch.nn.functional.normalize(class_means)
            elif 'vit' in cfg.trainer.model:
                global_model.model.head.weight.data = torch.nn.functional.normalize(class_means)
            else:
                global_model.model.classifier[2].weight.data = torch.nn.functional.normalize(class_means)
            
            test_accuracy, test_loss = inference(global_model.model, testloader, device)
            print('FedCOF acc on inference: ', test_accuracy)
            if cfg.wandb.activate and cfg.trainer.pretrained:
                wandb.log({'init acc': test_accuracy*100})

        if cfg.trainer.fed3r:
            class_means = []
            gram_mat_rr += cov_mats  # for Fed3R
            gram_mat = shrink_cov(gram_mat_rr, device, 0.01)
            
            inv_gram_mat = torch.linalg.pinv(gram_mat).float().to(device)
            for idx in range(cfg.dataset.num_classes):
                class_means.append(torch.matmul(inv_gram_mat, total_class_sums[idx]))
            class_means = torch.stack(class_means)

            if 'resnet' in cfg.trainer.model:
                global_model.model.fc.weight.data = torch.nn.functional.normalize(class_means)
            elif 'vit' in cfg.trainer.model:
                global_model.model.head.weight.data = torch.nn.functional.normalize(class_means)
            else:
                global_model.model.classifier[2].weight.data = torch.nn.functional.normalize(class_means)
            
            test_accuracy, test_loss = inference(global_model.model, testloader, device)
            print('Fed3R acc on inference: ', test_accuracy)
            if cfg.wandb.activate and cfg.trainer.pretrained:
                wandb.log({'init acc': test_accuracy*100})


    print("Before training")
    final_model, eval_score = trainer.train(
        data_provider=data_provider,
        metrics_reporter=metrics_reporter,
        num_total_users=data_provider.num_train_users(),
        distributed_world_size=1,
    )

    # get test accuracy after phase1 training
    test_accuracy, test_loss = inference(final_model.model, testloader, device)
    print(f'\nFinal Phase1 Testing--> loss: {test_loss} accuracy: {test_accuracy}\n')

    if cfg.wandb.activate:
        wandb.log({"final_acc": test_accuracy*100})
    
    

def run(cfg: DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))
    main(cfg)


if __name__ == "__main__":
    cfg = fl_config_from_json(json_config)
    # update with commandline args if they are passed, otherwise defaults to hardcoded cfg file vars
    set_cfg_from_cl(cfg)
    # print(cfg)
    validata_dataset_params(cfg)
    run(cfg)
