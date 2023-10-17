import numpy as np
import torch

def PEHE(t, y, TE, y0_pred, y1_pred):
    TE_pred = (y1_pred - y0_pred)  
    return torch.mean((TE - TE_pred)**2).item()

def ATE(t, y, TE, y0_pred, y1_pred):
    TE = torch.mean(TE)  
    TE_pred = torch.mean(y1_pred - y0_pred)
    return torch.abs(TE-TE_pred).item()

def RMSE(t, y, TE, y0_pred, y1_pred):
    y_pred = t * y1_pred + (1 - t) * y0_pred
    return torch.sqrt(torch.mean((y - y_pred)**2)).item()

def compute_variances(te, assigned_clusters, n_clusters):
    within_var, across_var = [], []
    for idx in range(n_clusters):
        te_by_cluster = te[assigned_clusters == idx]
        if len(te_by_cluster)>0:
            within_var.append(torch.var(te_by_cluster).item())
            across_var.append(torch.mean(te_by_cluster).item())        
    within_var = np.mean(within_var)
    across_var = np.var(across_var)
    return within_var, across_var
