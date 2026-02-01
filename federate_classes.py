from abc import ABC, abstractmethod
import torch  
import numpy as np

class Server(ABC):
    def __init__(self, client_data, n_classes, feature_space_dim):
        self.client_data = client_data
        self.n_classes = n_classes
        self.feature_space_dim = feature_space_dim
          
    
    @abstractmethod    
    def aggregate(self):
        pass 


class FedNCMServer(Server):
    def __init__(self, client_data, n_classes, feature_space_dim):
        super().__init__(client_data, n_classes, feature_space_dim)
        self.class_means = None 

    def get_classifier_weights(self):
        init_W =  torch.nn.functional.normalize(self.class_means)
        return init_W 
    
    def aggregate(self):
        class_mean_sums = {c:torch.zeros(size=(self.feature_space_dim,)) for c in range(self.n_classes)}
        class_sample_counts = {c:0 for c in range(self.n_classes)}
        class_means_ = {c:torch.zeros(size=(self.feature_space_dim,)) for c in range(self.n_classes)}
        
        for user_id, user_data in self.client_data.items():
            class_means = user_data["class_means"]
            sample_counts = user_data["sample_per_mean"]
            
            for class_id, class_mean in class_means.items():
                class_mean_sums[class_id] += class_mean * float(sample_counts[class_id])
                class_sample_counts[class_id] += sample_counts[class_id]

        for class_id in class_mean_sums:
            if class_sample_counts[class_id] > 0:
                class_means_[class_id] = class_mean_sums[class_id] / class_sample_counts[class_id]
        
        sorted_class_means = dict(sorted(class_means_.items(), key=lambda item: item[0]))
        sorted_class_means_list = list(sorted_class_means.values())
        self.class_means = torch.stack(sorted_class_means_list)
        

class Fed3RServer(Server):    
    def __init__(self, client_data,  n_classes, feature_space_dim, _lambda):
        super().__init__(client_data, n_classes, feature_space_dim)
        self.ridge_solution =  None
        self._lambda = _lambda
        
    def shrink_cov(self, cov,  alpha=1):
        iden = torch.eye(cov.shape[0])
        cov_ = cov + (alpha*iden)
        return cov_
    
    def get_classifier_weights(self):
        init_W =  torch.nn.functional.normalize(self.ridge_solution)
        return init_W 
    
    def aggregate(self):
        G = torch.zeros(size=(self.feature_space_dim, self.feature_space_dim))
        class_sums = {c:torch.zeros(size=(self.feature_space_dim,)) for c in range(self.n_classes)}
        
        for user_id, user_data in self.client_data.items():
            current_G = user_data["G"]
            G += current_G
            for class_id, class_sum in user_data["B"].items():
                class_sums[class_id] += class_sum 
        
        inv_G = torch.linalg.pinv(self.shrink_cov(G, alpha=self._lambda)).float() 
        self.ridge_solution = []
        
        for label_idx in range(self.n_classes):
            self.ridge_solution.append(torch.matmul(inv_G.float(), class_sums[label_idx].float().unsqueeze(0).T))
        
        self.ridge_solution = torch.stack(self.ridge_solution).float().squeeze(2)
        
        
class FedCOFServer(Server):
    def __init__(self, client_data,  n_classes, feature_space_dim, gamma):
        super().__init__(client_data, n_classes, feature_space_dim)
        self.ridge_solution =  None
        self.gamma = gamma 
        
        self.num_total = 0
        
    def shrink_cov(self, cov, gamma=1):
        iden = torch.eye(cov.shape[0])
        cov_ = cov + (gamma*iden)
        return cov_
    
    def get_classifier_weights(self):
        init_W =  torch.nn.functional.normalize(self.ridge_solution)
        return init_W 

    def get_num_round(self):
        return self.num_total
    
    def aggregate(self):
        covs = {c:[] for c in range(self.n_classes)}
        class_mean_clients = {c:[] for c in range(self.n_classes)}
        class_mean_sums = {c:torch.zeros(size=(self.feature_space_dim,)) for c in range(self.n_classes)}
        class_counts = {c: [] for c in range(self.n_classes)}

        class_sample_counts = {c:0 for c in range(self.n_classes)}
        G = torch.zeros(size=(self.feature_space_dim, self.feature_space_dim))
        class_sums = {c:torch.zeros(size=(self.feature_space_dim,)) for c in range(self.n_classes)}
        num_x = 0

        for user_id, user_data in self.client_data.items():
            class_means = user_data["class_means"]
            sample_counts = user_data["sample_per_mean"]
            
            for class_id, class_mean in class_means.items():
                class_mean_sums[class_id] += class_mean * float(sample_counts[class_id])
                class_sample_counts[class_id] += sample_counts[class_id]
                class_mean_clients[class_id].append(class_mean)
                class_counts[class_id].append(float(sample_counts[class_id]))
        
        class_means = {class_id: class_mean_sums[class_id] / class_sample_counts[class_id]
                            for class_id in class_mean_sums}
        
        sorted_class_means = dict(sorted(class_means.items(), key=lambda item: item[0]))
        sorted_class_means_list = list(sorted_class_means.values())
        self.class_means = torch.stack(sorted_class_means_list)

        all_count = 0
        g_means = 0
        for class_id, class_mean in class_means.items():
            if class_sample_counts[class_id] > 0:
                class_mean_clients[class_id] = torch.stack(class_mean_clients[class_id])
                class_counts[class_id] = torch.tensor(class_counts[class_id])
                all_count += class_sample_counts[class_id]
                g_means += class_means[class_id]*class_sample_counts[class_id]

                mu_x = class_mean_clients[class_id] - class_means[class_id]
                mu_x *= torch.sqrt(class_counts[class_id])[:, None]
                inside = torch.mm(mu_x.T, mu_x)
                if len(class_mean_clients[class_id]) > 1:
                    delta = inside/(len(class_mean_clients[class_id])-1)
                else:
                    delta = inside/(len(class_mean_clients[class_id]))
                covs[class_id] = self.shrink_cov(delta, gamma=self.gamma)
                num_x += len(class_mean_clients[class_id])

        self.num_total = num_x 
        
        g_means = g_means/all_count
        
        t1 = all_count*torch.mm(g_means.unsqueeze(1), g_means.unsqueeze(0))

        est_gram = torch.zeros(self.feature_space_dim, self.feature_space_dim)
        for label_idx in range(self.n_classes):
            if class_sample_counts[label_idx] > 0:
                est_gram += (class_sample_counts[label_idx]-1)*covs[label_idx]   
        est_gram = est_gram + t1

        # inv_G = torch.linalg.pinv(self.shrink_cov(est_gram, gamma=0)).float() 
        inv_G = torch.linalg.pinv(est_gram).float()
        self.ridge_solution = []
        
        for label_idx in range(self.n_classes):
            self.ridge_solution.append(torch.matmul(inv_G.float(), class_mean_sums[label_idx].float().unsqueeze(0).T))
        
        self.ridge_solution = torch.stack(self.ridge_solution).float().squeeze(2)
        


        
class Client(ABC):
    def __init__(self,  user_id, user_features, user_labels):
        self.user_id = user_id
        self.user_features = user_features
        self.user_labels = user_labels 
        self.statistics = {} 

    @abstractmethod
    def compute(self):
        pass   
    
    def send(self):
        return self.statistics 
    
    def reset(self):
        self.statistics ={}
        self.user_id = None 
        self.user_features = None 
        self.user_labels= None 
    

class FedNCMClient(Client):
    def __init__(self,  user_id, user_features, user_labels):
        super().__init__(user_id, user_features, user_labels)
        self.statistics = {"class_means":None, "sample_per_mean":None}
    
    def compute_class_means(self, labels, data):
        unique_classes = torch.unique(labels)

        # Dictionary to store the means and sample counts for each class
        class_means = {}
        class_sample_counts = {}

        # Loop over each unique class and compute the mean and sample count
        for class_label in unique_classes:
            # Get the data points corresponding to the current class
            class_data = data[labels == class_label]
            
            # Compute the mean for the class and the number of samples
            class_mean = class_data.mean(dim=0)
            class_sample_count = class_data.size(0)
            
            # Store the mean and sample count in the dictionary
            class_means[class_label.item()] = class_mean
            class_sample_counts[class_label.item()] = class_sample_count
            
        return class_means, class_sample_counts
        
        
    def compute(self):
        class_means, class_sample_counts = self.compute_class_means(self.user_labels, self.user_features)
        self.statistics["class_means"] = class_means
        self.statistics["sample_per_mean"] = class_sample_counts
 
        
class FedCOFClient(Client):
    
    def __init__(self, user_id, user_features, user_labels):
        super().__init__( user_id, user_features, user_labels)
        self.statistics = {"class_means":None, "sample_per_mean":None}
    
    def compute_class_means(self, labels, data):
        unique_classes = torch.unique(labels)

        # Dictionary to store the means and sample counts for each class
        class_means = {}
        class_sample_counts = {}

        # Loop over each unique class and compute the mean and sample count
        for class_label in unique_classes:
            # Get the data points corresponding to the current class
            class_data = data[labels == class_label]
            
            # Compute the mean for the class and the number of samples
            class_mean = class_data.mean(dim=0)
            class_sample_count = class_data.size(0)
        
            # Store the mean and sample count in the dictionary
            class_means[class_label.item()] = class_mean
            class_sample_counts[class_label.item()] = class_sample_count
            if class_sample_count > 0:
                if class_sample_count == 1:
                    feat_gen = []
                    x_ = class_data[0]
                    feat_gen.append(x_)
                    feat_gen.append(x_+(0.01**0.5)*torch.randn(x_.shape))
                    feat_gen = torch.cat(feat_gen, dim=0)
    
        return class_means, class_sample_counts
        

    def compute(self):
        class_means, class_sample_counts = self.compute_class_means(self.user_labels, self.user_features)
        self.statistics["class_means"] = class_means
        self.statistics["sample_per_mean"] = class_sample_counts
         

        
class Fed3RClient(Client):
    def __init__(self,  user_id, user_features, user_labels):
        super().__init__(user_id, user_features, user_labels)
        self.statistics = {"G":None, "B":None}
    
    def compute_G_B(self, data, labels):
        G = torch.mm(data.T, data)
        
        unique_classes = torch.unique(labels)

        # Dictionary to store the means and sample counts for each class
        sum_means = {}
 
        # Loop over each unique class and compute the mean and sample count
        for class_label in unique_classes:
            # Get the data points corresponding to the current class
            class_data = data[labels == class_label]
            
            # Compute the mean for the class and the number of samples
            class_mean = class_data.mean(dim=0)
            class_sample_count = class_data.size(0)
            sum_means[class_label.item()] = class_mean * class_sample_count 
             
        return G, sum_means
        
    def compute(self):
        self.statistics["G"], self.statistics["B"] = self.compute_G_B(self.user_features, self.user_labels)