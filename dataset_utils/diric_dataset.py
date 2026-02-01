import numpy as np 
import torch

# Partition function using Dirichlet distribution
def old_partition_dataset_by_dirichlet(dataset, num_clients, alpha):
    # Get labels for all data points
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    unique_labels = np.unique(labels)
    label_indices = {label: np.where(labels == label)[0] for label in unique_labels}
    client_indices = {i: [] for i in range(num_clients)}

    for label in unique_labels:
        indices = label_indices[label]
        np.random.shuffle(indices)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (proportions * len(indices)).astype(int)
        # Adjust for rounding errors
        proportions[-1] = len(indices) - proportions[:-1].sum()
        idx = 0
        for client_id, num_samples in enumerate(proportions):
            client_indices[client_id].extend(indices[idx: idx + num_samples])
            idx += num_samples

    return client_indices


def partition_dataset_by_dirichlet(dataset, num_clients, alpha):
    """
    Splits dataset samples among clients according to a Dirichlet distribution parameterized by alpha,
    and returns a dictionary with client IDs as keys and sample indices as values.
 
    Returns:
        Dict[int, List[int]]: A dictionary where keys are client IDs and values are lists of sample indices.
    """
    print(f"Distributing data using Dirichlet distribution with alpha={alpha}")
    
    # Extract labels from the dataset
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)

    # Initialize data structures
   
    multinomial_vals = []
    examples_per_label = []

    # Count the number of examples for each class
    for label in unique_labels:
        examples_per_label.append(int(np.argwhere(labels == label).shape[0]))


    # Each client has a multinomial distribution over classes drawn from a Dirichlet
    for i in range(num_clients):
        proportion = np.random.dirichlet(alpha * np.ones(num_classes))
        multinomial_vals.append(proportion)
        
        
    multinomial_vals = np.array(multinomial_vals)
    example_indices = []

    # Shuffle and store indices for each class
    for label in unique_labels:
        label_k = np.where(labels == label)[0]
        np.random.shuffle(label_k)
        example_indices.append(label_k)
        
    example_indices = np.array(example_indices, dtype=object)
    
    client_indices = {i:[] for i in range(num_clients)}
    count = np.zeros(num_classes).astype(int)
    
    
    examples_per_client = int(labels.shape[0] / num_clients)

    max_fail = 0
    
    for k in range(num_clients):
        for i in range(examples_per_client):
            sampled_label = np.argwhere(np.random.multinomial(1, multinomial_vals[k, :]) == 1)[0][0]

            label_indices = example_indices[sampled_label]

            # very sketchy work around for low dataset sample sizes
            try:
                client_indices[k].append(label_indices[count[sampled_label]])
                
            except IndexError:
                i -= 1
                max_fail += 1
                if max_fail >1000:
                    print('(line 142) Dirichlet Sharder too many fails')
                    exit(1)
                continue

            count[sampled_label] += 1
            if count[sampled_label] == examples_per_label[sampled_label]:
                multinomial_vals[:, sampled_label] = 0
                multinomial_vals = (
                        multinomial_vals /
                        multinomial_vals.sum(axis=1)[:,None])

    return client_indices



# Custom dataset class that includes client IDs
class ClientDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, client_id, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.client_id = client_id
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        img, label = self.dataset[actual_idx]
        if self.transform:
            img = self.transform(img)
        return img, label, self.client_id