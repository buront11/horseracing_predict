from tqdm import tqdm

import lightgbm as lgb
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.dataloading import GraphDataLoader

def train(dataset, model, criterion, optim, batch_size, epochs, device):

    train_dataloader = GraphDataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True)

    model = model.to(device)
    criterion = criterion
    optim = optim

    min_loss = 1e9

    model.train()

    for epoch in tqdm(range(epochs)):
        running_loss = 0
        total = 0
        correct = 0
        for batched_graph, labels in train_dataloader:
            optim.zero_grad()

            feats = batched_graph.ndata['feat'].to(device)
            outputs = model(batched_graph, feats)
            labels = labels.to(device)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)

            loss.backward()
            optim.step()

        print('epoch :{}'.format(epoch+1))
        print('loss     :{}'.format(running_loss))

        if min_loss >= running_loss:
            model_path = './data/weight'
            torch.save(model.state_dict(), model_path)


def node_train(dataset, model, criterion, optim, batch_size, epochs, device):
    train_dataloader = GraphDataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True)

    model = model.to(device)
    criterion = criterion
    optim = optim

    min_loss = 1e9

    model.train()

    for epoch in tqdm(range(epochs)):
        running_loss = 0
        total = 0
        correct = 0
        for batched_graph in train_dataloader:
            optim.zero_grad()

            feats = batched_graph.ndata['feat'].to(device)
            labels = batched_graph.ndata['label'].to(device)
            outputs = model(batched_graph, feats)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)

            loss.backward()
            optim.step()

        print('epoch :{}'.format(epoch+1))
        print('loss     :{}'.format(running_loss))

        if min_loss >= running_loss:
            model_path = './data/weight'
            torch.save(model.state_dict(), model_path)

def light_gbm_train():
    pass