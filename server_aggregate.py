import torch
import random
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models.squeezenet import SqueezeNet1_1_Weights
 
import argparse
import wandb
import csv 
import pickle 
from torchvision.models import mobilenet_v2
from torchvision.models import squeezenet1_1,  MobileNet_V2_Weights
import timm
import os   
import pandas as pd  
import sys 
from federate_classes import * 
import shutil
import string

def experiment_folder(root_path, dev_mode, approach_name):
    if os.path.exists(os.path.join(root_path, 'exp_folder')):
        shutil.rmtree(os.path.join(root_path, 'exp_folder'), ignore_errors=True)

    if dev_mode:
        exp_folder = 'exp_folder'
    else:
        exp_folder = approach_name + '_' + ''.join(random.choices(string.ascii_letters + string.digits, k=8))

    out_path = os.path.join(root_path, exp_folder)
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    return out_path, exp_folder

def store_results_to_csv(test_accuracy, args, out_path):
    results = []

    # Add this after computing the test accuracy
    results.append({
        "Approach": args.approach,
        "Model Type": args.model_type,
        "Dataset": args.dataset,
        "Number of Clients": args.num_clients,
        "Seed": args.seed,
        "Seed to Load": args.seed_to_load,
        "Test Accuracy": test_accuracy,
        "Lambda": args.lamb if args.approach in ["fed3r"] else None,
        "Gamma Shrink": args.gamma_shrink if args.approach == "fedcof" else None,
    })
    
    # Write results to CSV at the end of the script
    csv_file = os.path.join(out_path, "summary.csv")

    # Write header if file doesn't exist, otherwise append results
    write_header = not os.path.exists(csv_file)
    with open(csv_file, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        if write_header:
            writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to {csv_file}")

    
def get_args():
    parser = argparse.ArgumentParser()

    """
    Structural hyperparams 
    """
    parser.add_argument("--dataset", type=str,default="inat", choices=["inat", "cifar100", "imagenet-r",
                                                                        "cars", "cub"])
    
    parser.add_argument("--model_type",type=str, default="squeezenet", choices=["squeezenet", 
                                                                                "mobilenet",
                                                                                "vit"])
    
    parser.add_argument("--num_clients", type=int, default=100, help="number of clients to load for the specific dataset")
    parser.add_argument("--alpha", type=float, default=0.1, help="(for loading alpha for dirichlet distribution used to generate the clients")
    parser.add_argument("--seed_to_load", type=int, default=0, help="seed to load the clients")
    
    parser.add_argument("--seed", type=int, default=0, help="current seed to use for the experiment")
    parser.add_argument("--src_path", type=str, default="new_federated_features")
    parser.add_argument("--approach",type=str, default="fedcof", choices=["fedncm","fed3r","fedcof"])
    parser.add_argument("--initialization_type", type=str, choices=["single","multiple_without_replacement"],default="single")
    parser.add_argument("--clients_per_round", type=int, default=10, help="how many clients sample for initialization")
    parser.add_argument("--wandb", type=int, default=0, help="activate wandb")
    parser.add_argument("--run_name", type=str, help="name of run in wandb")
    parser.add_argument("--wandb_project", type=str, default="FedL-new", help="name of project in wandb")
    parser.add_argument("--wandb_entity", type=str, default="dgoswami", help="name of entity in wandb")

    
    # FEDCOF and Fed3R hyperparams
    parser.add_argument("--lamb",type=float, default=0.01, help="lambda for ridge regression solution in Fed3R")
    parser.add_argument("--gamma_shrink",type=float, default=1, help="gamma for shrinkage in FedCOF")
 
    parser.add_argument("--outpath", "-op",default="./output_folder", type=str) 
     
    args = parser.parse_args()

    non_default_args = {
            opt.dest: getattr(args, opt.dest)
            for opt in parser._option_string_actions.values()
            if hasattr(args, opt.dest) and opt.default != getattr(args, opt.dest)
    }

    default_args = {
            opt.dest: opt.default
            for opt in parser._option_string_actions.values()
            if hasattr(args, opt.dest)
    }

    all_args = vars(args)    
    
    return args, all_args, non_default_args, default_args


def wandb_setup(cfg):
    if cfg.run_name:
        os.environ['WANDB_NAME'] = cfg.run_name
        os.environ['WANDB_START_METHOD'] = "thread"

    # need to set wandb run_dir to something we can access to avoid permission denied error.
    # See https://github.com/wandb/client/issues/3437
    wandb_path = f'scratch/{os.environ.get("USER","username")}/wandb'
    #wandb_path = f'wandb'
    if not os.path.isdir(wandb_path):
        os.makedirs(wandb_path, mode=0o755, exist_ok=True)

    # if using wandb check project and entity are set
    assert not cfg.wandb_project == '' and not cfg.wandb_entity == ''
    
    wandb.login()
    wandb.init(dir=wandb_path, project=cfg.wandb_project, entity=cfg.wandb_entity)
    general_args = {
        "seed": cfg.seed,
        "algorithm": cfg.approach,
        "dataset": cfg.dataset,
        "model_type": cfg.model_type,
    }



if __name__ == "__main__":
    args, all_args, non_default_args, all_default_args = get_args()
    if args.dataset == "inat":
        args.num_clients = 9275
        
    print("Approach: ", args.approach)
    print("Model Type: ", args.model_type)
    print("Dataset: ", args.dataset)
    print("Number of Clients: ", args.num_clients)
    print("Seed: ", args.seed)
    print("Seed to Load: ", args.seed_to_load)
    if args.approach == "fed3r":
        print("Hyperparams Fed3R: lambda {}", args.lamb)
    elif args.approach == "fedcof":
        print("Hyperparams FedCOF: gamma_shrink {}".format(args.gamma_shrink))

        
    cuda_enabled = torch.cuda.is_available()
    device = torch.device(f"cuda:{0}" if cuda_enabled else "cpu")
    seed = args.seed
    print(device)
    
    
    """
    Organize the results in output folders 
    """
    dev_mode = False
    if not os.path.exists(args.outpath):
        os.mkdir(args.outpath)
        
    out_path, exp_name = experiment_folder(args.outpath, dev_mode, args.approach)
    

    if cuda_enabled:
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    else:
        torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


    dataset = args.dataset
    model_type = args.model_type 
    approach = args.approach 
    
    
    if dataset == 'inat':
        num_classes = 1203
        number_of_clients = 9275
    elif dataset == "cifar100":
        num_classes = 100
        number_of_clients = args.num_clients
    elif dataset == "imagenet-r":
        num_classes = 200
        number_of_clients = args.num_clients
    elif dataset == "cars":
        num_classes = 196
        number_of_clients = args.num_clients
    elif dataset == "cub":
        num_classes = 200
        number_of_clients = args.num_clients
    else:
        sys.exit("Dataset not configurated")
        
    "Train Features for classifier initialization"
    
    
    print("Instating Model {}".format(model_type))
    if model_type == "squeezenet":
        feat_dims = 512
        model = squeezenet1_1(weights=SqueezeNet1_1_Weights.DEFAULT)
        model.classifier = torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(512, num_classes, bias=False) # 
        )
    elif model_type == "mobilenet":
        feat_dims = 1280 
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    
        model.classifier = torch.nn.Sequential(
                # nn.Dropout(p=dropout),
                torch.nn.Identity(),
                torch.nn.Identity(),
                torch.nn.Linear(1280, num_classes, bias=False),
            )
    elif model_type == "vit":
        feat_dims = 768
        model = timm.create_model("vit_base_patch16_224_in21k", pretrained=True)
        model.head = torch.nn.Linear(768, num_classes, bias=False)

        
    lp = True
    if lp:
        for name, param in model.named_parameters():
            param.requires_grad = False
    
    if args.wandb == 1:
        run_dir = f'scratch/{os.environ.get("USER", "username")}/{args.run_name}'  # add your user name
        if not os.path.isdir(run_dir):
            os.makedirs(run_dir, mode=0o755, exist_ok=True)
        wandb_setup(args)
        wandb.watch(model, log_freq=1)
 
    embedder = None 
    print("Inference on the dataset to compute Weight Matrix Initialization")

    
    #{"user":{"data":tensor,"labels":tensor},..}
    if args.dataset in ["inat"]:
        with open(os.path.join(args.src_path, 
                            "{}_feats".format(args.dataset,
                                                            str(args.alpha),
                                                            str(args.num_clients)),
                            model_type,
                            "seed_{}".format(str(args.seed_to_load)),
                            "train_client_data.pt"), 'rb') as f:
            
            train_client_dict = pickle.load(f)
            
        # Instance test features for computing accuracy
        with open(os.path.join(args.src_path, 
                               "{}_feats".format(args.dataset),
                                model_type,
                                "seed_{}".format(str(args.seed_to_load)),
                                "test_data.pt"), 'rb') as f:
            
            test_dict = pickle.load(f)

    else:
        with open(os.path.join(args.src_path, 
                               "{}_alpha_{}_K_{}_feats".format(args.dataset,
                                                               str(args.alpha),
                                                               str(args.num_clients)),
                                model_type,
                                "seed_{}".format(str(args.seed_to_load)),
                                "train_client_data.pt"), 'rb') as f:
            
            train_client_dict = pickle.load(f)
        
        with open(os.path.join(args.src_path, 
                               "{}_alpha_{}_K_{}_feats".format(args.dataset,
                                                               str(args.alpha),
                                                               str(args.num_clients)),
                                model_type,
                                "seed_{}".format(str(args.seed_to_load)),
                                "test_data.pt"), 'rb') as f:
            
            test_dict = pickle.load(f)
            
            
    if approach == "fedncm":
        class_name_client = "FedNCMClient"
        class_name_server = "FedNCMServer"
    
    elif approach == "fed3r":
        class_name_client = "Fed3RClient"
        class_name_server = "Fed3RServer"
        
    elif approach == "fedcof":
        class_name_client = "FedCOFClient" 
        class_name_server = "FedCOFServer"
    
    # Initialize a client according the name of an approach     
    ClientClass = globals()[class_name_client]
    
    server_data = {}
    num = 0
    num_old = 0
    print(len(train_client_dict))

    for user_id, data_dict in train_client_dict.items():
        # print(user_id)
        current_feat = data_dict["data"]
        current_labels = data_dict["labels"]
        
        client_instance = ClientClass(user_id, current_feat, current_labels)
            
        client_instance.compute()
        
        client_statistics = client_instance.send()
        # reset client
        client_instance.reset()
        client_instance = None
        server_data[user_id] = client_statistics
        num += 1

        # print('Clients seen so far: ',num)
        if num >0  and (num == number_of_clients):  #num % 30 == 0 or
            print('Clients seen so far: ',num)
            # Initialize a server according the name of an approach  
            ServerClass = globals()[class_name_server]
            # Now the server must aggregate the client statistics for classifier initialization 
            if approach == "fedncm":
                server_instance = ServerClass(server_data, num_classes, feat_dims)
            elif approach == "fed3r":
                server_instance = ServerClass(server_data, num_classes, feat_dims, args.lamb)
            elif approach == "fedcof":
                server_instance = ServerClass(server_data, num_classes, feat_dims, args.gamma_shrink)
            server_instance.aggregate()
            optimal_weights = server_instance.get_classifier_weights()
            # num_round = server_instance.get_num_round()
            
            if model_type == "vit":
                model.head.weight.data = optimal_weights
            else:
                model.classifier[2].weight.data = optimal_weights
            model = model.to(device)
            model.eval()
    
            test_dataset = TensorDataset(test_dict["data"], test_dict["labels"])
            test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
            with torch.no_grad():
                correct, total = 0, 0
                for i, (features, targets) in enumerate(test_loader):
                        features = features.to(device)
                        if model_type == "vit":
                            outputs = model.head(features)
                        else:
                            outputs = model.classifier[-1](features)
                        
                        predicts = torch.max(outputs, dim=1)[1]
                        correct += (predicts.cpu() == targets).sum()
                        total += len(targets)

                test_accuracy = np.around(correct.item() * 100 / total, decimals=2)
            
            print("Approach {} Test Accuracy {}".format(approach, test_accuracy))
            store_results_to_csv(test_accuracy, args, out_path)
            
            with open(os.path.join(out_path, "server_stats.pkl"), "wb") as f:
                    pickle.dump(server_data, f)

            if args.wandb == 1:
                wandb.log({'initial acc': test_accuracy})
