# -*- coding: utf-8 -*-      
#--------------------------------------------
# Author:chenhao
# Date:2020-12-27 14:35
# Description:  
#--------------------------------------------
import torch
from src.metric.roc_auc import roc_auc_score

def evaluate(model, test_iter, device):
    predicts = []
    labels = []
    with torch.no_grad():
        for batch in test_iter:
            labels.append(batch['label'].to(device).float())
            batch = {t: {k: v.to(device) for k, v in d.items()} for t, d in batch.items() if isinstance(d, dict)}

            predict = model(batch)
            predicts.append(predict)
    labels = torch.cat(labels)
    predicts = torch.cat(predicts)

    auc = roc_auc_score(labels, predicts)

    return auc
