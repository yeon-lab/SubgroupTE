import torch
import torch.nn as nn
from torch.nn import Parameter

SEED = 1111
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class ClusterAssignment(nn.Module):
    def __init__(
        self,
        n_clusters,
        init = 'kmeans++',
        w = 1
    ):
        super(ClusterAssignment, self).__init__()
        self.n_clusters = n_clusters
        self.init = init
        self.w = w
        self.cluster_centers = None
        self.eps = 1e-5
            
    def forward(self, input):
        if self.cluster_centers is None:
            return torch.zeros(len(input), self.n_clusters).to(input.device)
        clusters = torch.cdist(input, self.cluster_centers)
        clusters = nn.functional.softmin(clusters, dim=1)
        return clusters

    def kernel_density_estimate(self, input, centroid, bandwidth=0.5):
        kernel = torch.exp(-(input - centroid).pow(2).sum(dim=1) / (2 * bandwidth ** 2)) + self.eps
        kernel /= kernel.sum()
        return (kernel.unsqueeze(1) * (input - centroid)).sum(dim=0)

    def _adjust_centers(self, input): 
        if self.cluster_centers is None:
            self._centroid_init(input)
        kde = torch.stack(
            [self.kernel_density_estimate(input, centroid) for centroid in self.cluster_centers.data])
        self.cluster_centers.data += kde 
        
    def _update_centers(self, input, cluster_out):
        for idx in range(self.n_clusters):
            filtered = input[cluster_out==idx]
            if len(filtered) > 0:
                self.cluster_centers.data[idx] = (1-self.w) * self.cluster_centers.data[idx] + \
                                                                    self.w * torch.mean(filtered, 0)  
    def _centroid_init(self, input):
        self.cluster_centers = Parameter(torch.zeros(
                   self.n_clusters, input.shape[1])).to(input.device)
        if self.init == 'kmeans++':
            self._kmeans_plusplus_init(input)  
        else:
            self._random_init(input) 
                
    def _random_init(self, input):
        indices = torch.randperm(input.shape[0])[:self.n_clusters]
        for idx in range(self.n_clusters):
            self.cluster_centers.data[idx] = input[indices[idx]]

    def _kmeans_plusplus_init(self, input):
        #self.cluster_centers.data[0] = input[torch.randint(0, input.shape[0], (1,))]
        self.cluster_centers.data[0] = input[0]
        for idx in range(1, self.n_clusters):
            distances = torch.cdist(input, self.cluster_centers.data[:idx]).min(dim=1).values 
            probabilities = distances / torch.sum(distances)
            centroid_index = torch.multinomial(probabilities, 1)
            self.cluster_centers.data[idx] = input[centroid_index]

    

    
   
