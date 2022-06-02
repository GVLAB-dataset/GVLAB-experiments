import torch
from torch import nn

from models.gvlab_backend import BackendModel


class BaselineModel(nn.Module):

    def __init__(self, backend_model):
        super(BaselineModel, self).__init__()
        self.backend_model = backend_model
        pair_embed_dim = self.backend_model.text_dim + self.backend_model.image_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(pair_embed_dim),
            nn.Linear(pair_embed_dim, int(pair_embed_dim/4)), # for ViT-B/32 it will be 256
            nn.ReLU(),
            nn.Linear(int(pair_embed_dim/4), 1)
        )

    def forward(self, input_image_vector, text_vector):

        x = torch.cat([input_image_vector[0], text_vector[0]], dim=1) # what to do here ?

        x = self.classifier(x)
        return x


