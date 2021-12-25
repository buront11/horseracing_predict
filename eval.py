import torch
from torch._C import device
import torch.nn.functional as F

from models import GATClassifier

from datasets import HorceDataset
from dgl.dataloading import GraphDataLoader

def eval():
    device = 'cuda'
    model = GATClassifier().to(device)

    model_path = './data/weight'
    model.load_state_dict(torch.load(model_path))

    model.eval()

    dataset = HorceDataset('test')

    print(dataset[0])

    test_dataloader = GraphDataLoader(dataset, batch_size=1, shuffle=False)
    print(test_dataloader)

    
    for g, l in test_dataloader:
        feat = g.ndata['feat'].to(device)

        outputs = model(g, feat)
        _, pred = torch.max(outputs.data, 1)

        print(pred)
        print(l)

if __name__=='__main__':
    eval()