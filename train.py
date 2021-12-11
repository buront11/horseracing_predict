from tqdm import tqdm

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.dataloading import GraphDataLoader

def train(dataset, model, criterion, optim, batch_size, epochs, device):

    train_dataloader = GraphDataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True)    

    model = model.to(device)
    criterion = criterion
    optim = optim

    for epoch in tqdm(range(epochs)):
        for batched_graph, labels in train_dataloader:
            optim.zero_grad()

            feats = batched_graph.ndata['feat']
            outputs = model(batched_graph, feats)
            loss = criterion(outputs, labels)

            loss.backward()
            optim.step()