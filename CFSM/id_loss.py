import os
import torch
from torch import nn
from model_irse import Backbone

class IDLoss(nn.Module):
    def __init__(self, device='cuda', ckpt_dict=None):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se').to(device)
        self.facenet.load_state_dict(torch.load(ckpt_dict, map_location=torch.device('cpu')))
        self.facenet.eval()

    def extract_feats(self, x):
        #with torch.no_grad():
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, x, y):
        n_samples = x.shape[0]
        x_feats = self.extract_feats(x)
        y_feats = self.extract_feats(y)

        id_loss = 1-torch.matmul(x_feats.unsqueeze(1), y_feats.unsqueeze(-1)).squeeze(-1)
        #id_loss = id_loss/2

        return id_loss
