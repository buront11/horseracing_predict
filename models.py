import dgl
import dgl.nn.pytorch as dglnn
import torch.nn as nn
import torch.nn.functional as F

class GATClassifier(nn.Module):
    # 馬の最大頭数は18頭なので分類は18classとする
    def __init__(self, in_feat=143, hidden_feat=256, n_classifier=18):
        super(GATClassifier, self).__init__()
        self.conv1 = dglnn.GATConv(in_feat, hidden_feat)
        self.conv2 = dglnn.GATConv(hidden_feat, hidden_feat)
        self.fc1 = nn.Linear(hidden_feat, n_classifier)

    def forward(self, g, h):
        # Apply graph convolution and activation.
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        with g.local_scope():
            g.ndata['h'] = h
            # Calculate graph representation by average readout.
            hg = dgl.mean_nodes(g, 'h')
            x = self.fc1(hg)
            x = F.softmax(x)
            return x