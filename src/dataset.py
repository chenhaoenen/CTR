# -*- coding: utf-8 -*-      
#--------------------------------------------
# Author:chenhao
# Date:2021-01-02 17:07
# Description:  
#--------------------------------------------
import json
from torch.utils.data import Dataset, DataLoader

class SubsetDataset(Dataset):
    def __init__(self, file):
        self.data = []
        with open(file, 'r') as fr:
            line = fr.readline()
            while line:
                feat = {}; sparse = {}; dense = {}
                dic = json.loads(line.strip())
                for key in dic:
                    if '_idx' in key:
                        sparse[key] = dic[key]
                    elif key == 'click':
                        feat['label'] = dic[key]
                    else:
                        dense[key] = dic[key]
                feat['sparse'] = sparse
                feat['dense'] = dense
                self.data.append(feat)
                line = fr.readline()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def subsetDataloader(path, batch_size, worker_init, num_workers=1):
    train_dataset = SubsetDataset(path)
    return DataLoader(dataset=train_dataset,
                      batch_size=batch_size,
                      num_workers=num_workers,
                      worker_init_fn=worker_init,
                      pin_memory=True,
                      shuffle=True)