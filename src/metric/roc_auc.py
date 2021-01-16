# -*- coding: utf-8 -*-      
#--------------------------------------------
# Author:chenhao
# Date:2020-12-27 15:10
# Description:  
#--------------------------------------------
import torch

def roc_auc_score(y_true, y_score):
    """ROC AUC score in PyTorch
    Args:
        y_true (Tensor): [B]
        y_score (Tensor): [B]
    """
    device = y_score.device
    y_true.squeeze_()
    y_score.squeeze_()
    if y_true.shape != y_score.shape:
        raise TypeError(f"Shape of y_true and y_score must match. Got {y_true.shape()} and {y_score.shape()}.")

    desc_score_indices = torch.argsort(y_score, descending=True)
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    distinct_value_indices = torch.nonzero(y_score[1:] - y_score[:-1], as_tuple=False).squeeze()
    threshold_idxs = torch.cat([distinct_value_indices, torch.tensor([y_true.numel() - 1], device=device)])

    tps = torch.cumsum(y_true, dim=0)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    tps = torch.cat([torch.zeros(1, device=device), tps])
    fps = torch.cat([torch.zeros(1, device=device), fps])

    fpr = fps / fps[-1]
    tpr = tps / tps[-1]

    area = torch.trapz(tpr, fpr).item()

    return area

