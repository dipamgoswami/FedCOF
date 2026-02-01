import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader 
from torchvision.models.squeezenet import SqueezeNet1_1_Weights
import torch.nn.functional as F
from torchvision import transforms
import argparse
from tqdm import tqdm 
 
import pickle 
from torchvision.models import mobilenet_v2
from torchvision.models import squeezenet1_1,  MobileNet_V2_Weights
import timm
import os   
from dataset_utils.inaturalist_dataset import INaturalistDataset
from dataset_utils.cub_dataset import Cub2011
from dataset_utils.standford_cars import StanfordCars

from dataset_utils.diric_dataset import ClientDataset
from dataset_utils.diric_dataset import partition_dataset_by_dirichlet
import torchvision
 
from torch.utils.data import ConcatDataset
  

def get_args():
    parser = argparse.ArgumentParser()

    """
    Structural hyperparams 
    """
    parser.add_argument("--dataset", type=str,default="cars", choices=["inat", "cifar100", "imagenet-r",
                                                                            "cars", "cub"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model_type",type=str, default="mobilenet", choices=["squeezenet", 
                                                                                "mobilenet",
                                                                                "vit"])
    parser.add_argument("--dst_path", type=str, default="new_federated_features")
    
    parser.add_argument("--dataset_path", type=str, default="Dataset")
    
    parser.add_argument("--num_clients", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=12)   
  
  
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


if __name__ == "__main__":
    
    args, all_args, non_default_args, all_default_args = get_args()
    cuda_enabled = torch.cuda.is_available()
    device = torch.device(f"cuda:{0}" if cuda_enabled else "cpu")
    seed = args.seed
    print(device)

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
    
    if not os.path.exists(args.dst_path):
        os.mkdir(args.dst_path)
        

    dataset = args.dataset
    model_type = args.model_type 
 
    if dataset == 'inat':
        print("Inaturalist")
        folder_name_dataset = "inat"
        num_classes = 1203
        use_path = True
        train_transform = transforms.Compose(
            [   
                transforms.Resize(299),
                transforms.CenterCrop((224, 224)), 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
 
        test_transform = transforms.Compose(
            [   transforms.Resize(299),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        training_dataset = INaturalistDataset(images_root_dir=os.path.join(args.dataset_path,
                                                                           folder_name_dataset,
                                                                           "train_val_images"),
                                           csv_file=os.path.join(args.dataset_path,
                                                                folder_name_dataset,
                                                                "inaturalist-user-120k",
                                                                "federated_train_user_120k.csv"),
                                           transform=train_transform)
   
        
        testing_dataset =  INaturalistDataset(images_root_dir=os.path.join(args.dataset_path,
                                                                           folder_name_dataset,
                                                                           "train_val_images"),
                                              csv_file=os.path.join(args.dataset_path,
                                                                    folder_name_dataset,
                                                                    "inaturalist-user-120k","test.csv"),
                                              transform=test_transform)
        
    else:
        
        if dataset == "cifar100":
            print("CIFAR-100 Dataset")
            num_classes = 100
            use_path = False
            train_transform = transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.491399689874, 0.482158419622, 0.446530924224), (0.247032237587, 0.243485133253, 0.261587846975))
                ]
            )

            test_transform = transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.491399689874, 0.482158419622, 0.446530924224),
                                        (0.247032237587, 0.243485133253, 0.261587846975))
                ]
            )

            # Load CIFAR-100 dataset
            train_dataset_full = torchvision.datasets.CIFAR100(root=args.dataset_path, train=True, download=True)
            testing_dataset = torchvision.datasets.CIFAR100(root=args.dataset_path, train=False, download=True, transform=test_transform)

        elif dataset == "imagenet-r":
            print("ImageNet-R Dataset")
            folder_name_dataset = "imagenet-r"
            num_classes = 200   
            use_path = False
            train_transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            test_transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

            # Load ImageNet-R dataset
            train_dataset_full = torchvision.datasets.ImageFolder(root=os.path.join(args.dataset_path,
                                                                                    folder_name_dataset,
                                                                                    "train"))
            
            testing_dataset = torchvision.datasets.ImageFolder(root=os.path.join(args.dataset_path, 
                                                                              folder_name_dataset,
                                                                              "test"), transform=test_transform)

        
        elif dataset == "cub":
            print("CUB-200-2011 Dataset")
            folder_name_dataset = "CUB_200_2011"
            num_classes = 200
            use_path = False
            train_transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            test_transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

            # Load CUB dataset
            train_dataset_full = Cub2011(root=os.path.join(args.dataset_path,
                                                           folder_name_dataset), train=True, download=False)
            testing_dataset = Cub2011(root=os.path.join(args.dataset_path,
                                                           folder_name_dataset), train=False, download=False, transform=test_transform)
           
        elif dataset == "cars":
            print("Stanford Cars Dataset")
            folder_name_dataset = "stanford_cars"
            num_classes = 196
            use_path = False
            train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            test_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
       
            train_dataset_full = StanfordCars(root=args.dataset_path, split= "train",
                                              download=False)
            
            testing_dataset = StanfordCars(root=args.dataset_path, split= "test", download=False, transform=test_transform)


        print("Partitioning the dataset {}  to the clients".format(dataset))
        ### Create a partition for Cifar-100, ImageNet-R, CUB, Cars employing the dirichlet distribution 
        client_indices = partition_dataset_by_dirichlet(train_dataset_full, args.num_clients, args.alpha)

        # Create client datasets
        client_datasets = []
        for client_id, indices in client_indices.items():
            client_datasets.append(ClientDataset(train_dataset_full, indices, client_id, transform=train_transform))

        # Concatenate all client datasets
        training_dataset = ConcatDataset(client_datasets)
        
            
    train_loader = DataLoader(training_dataset, batch_size=128, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(testing_dataset, batch_size=128, shuffle=True, num_workers=args.num_workers)    
    
    print("Extracting Features for Model {}".format(model_type))
    if model_type == "squeezenet":
        print("Squeezenet")
        feat_dims = 512
        model = squeezenet1_1(weights=SqueezeNet1_1_Weights.DEFAULT)
        model.classifier = torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(512, num_classes, bias=False) # 
        )
    elif model_type == "mobilenet":
        print("Mobilenet")
        feat_dims = 1280 
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    
        model.classifier = torch.nn.Sequential(
                # nn.Dropout(p=dropout),
                torch.nn.Identity(),
                torch.nn.Identity(),
                torch.nn.Linear(1280, num_classes, bias=False),
            )
    elif model_type == "vit":
        print("Vit")
        feat_dims = 768
        model = timm.create_model("vit_base_patch16_224_in21k", pretrained=True)
        model.head = torch.nn.Linear(768, num_classes, bias=False)
        
    lp = True
    if lp:
        for name, param in model.named_parameters():
            param.requires_grad = False
    
 
    embedder = None 
    print("Inference on the dataset to compute Weight Matrix Initialization")
    model = model.to(device)
    model.eval()
    
    # In train 
    # Iterate Over Client in Naturalist: save each client feature and label: inat_features/model/train_client_data.pt
    # Iterate Over Test in Naturalist: save all test_featurs: inat_features/model/test/test_data.pt
    
    if args.dataset in ["inat"]:
        dst_path = os.path.join(args.dst_path,"{}_feats".format(args.dataset))
    else:
        dst_path = os.path.join(args.dst_path,"{}_alpha_{}_K_{}_feats".format(args.dataset, 
                                                                                           args.alpha,
                                                                                           args.num_clients))
        
        
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    
    dst_path = os.path.join(dst_path, model_type) 
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
        
    # create a seed subfolder 
    
    dst_path = os.path.join(dst_path, "seed_{}".format(seed))
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    
    
    client_data = {}

    with torch.no_grad():
        for inputs, targets, users in tqdm(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Get embeddings from the model and process them
            if model_type == "vit":
                embs = model.forward_features(inputs)[:, 0, :]
            else:
                embs = model.features(inputs)
                embs = F.adaptive_avg_pool2d(embs, 1).squeeze(2).squeeze(2)
            
            # Iterate over each user in the current batch
            for i, user in enumerate(users):
                if torch.is_tensor(user):
                    user = str(user.item())
                    
                if user not in client_data:
                    client_data[user] = {
                        'data': [],
                        'labels': []
                    }
                
                # Append the corresponding embedding and label to the user's data
                client_data[user]['data'].append(embs[i].cpu())
                client_data[user]['labels'].append(targets[i].cpu().item())

    # Convert lists to tensors
    for user in client_data:
        client_data[user]['data'] = torch.stack(client_data[user]['data'])
        current_labels =  [label for label in client_data[user]['labels']]
        client_data[user]['labels'] = torch.tensor(current_labels)

 
    with open(os.path.join(dst_path,"train_client_data.pt"), 'wb') as f:
        pickle.dump(client_data, f)
    

    test_data = {"data":[],"labels":[]}
    with torch.no_grad():
        for inputs, targets  in tqdm(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Get embeddings from the model and process them
            if model_type == "vit":
                embs = model.forward_features(inputs)[:, 0, :]
            else:
                embs = model.features(inputs)
                embs = F.adaptive_avg_pool2d(embs, 1).squeeze(2).squeeze(2)
    
            # Append the corresponding embedding and label to the user's data
            test_data['data'].append(embs.cpu())
            test_data['labels'].append(targets.cpu())
    
    test_data['data'] = torch.cat(test_data['data'], dim=0)
    all_labels = [label for label in test_data['labels']]
    test_data['labels'] = torch.cat(all_labels)

    with open(os.path.join(dst_path,"test_data.pt"), 'wb') as f:
        pickle.dump(test_data, f)

    