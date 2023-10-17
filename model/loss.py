import torch
import torch.nn.functional as F


def SubgroupTE_loss(y, t, t_pred, y0_pred, y1_pred, y0_pred_init, y1_pred_init, alpha=1.0, beta=1.0, gamma=1.0):
    loss_t = F.binary_cross_entropy(t_pred, t)

    loss_y_init = torch.sum((1. - t) * torch.square(y - y0_pred_init)) + torch.sum(t * torch.square(y - y1_pred_init))
    loss_y = torch.sum((1. - t) * torch.square(y - y0_pred)) + torch.sum(t * torch.square(y - y1_pred))

    loss = alpha*loss_t + beta*loss_y_init + gamma*loss_y
    return loss
