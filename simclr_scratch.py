import torch.nn as nn
import torch
import torch.nn.functional as F

class SimCLR(nn.Module):
    def __init__(self, encoder, projection_dim=128, n_features=512):
        super(SimCLR, self).__init__()

        self.encoder = encoder
        self.projection_dim = projection_dim
        self.n_features = n_features

        self.projector = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )
    

    def forward(self, img1, img2):
        hi = self.encoder(img1)
        hj = self.encoder(img2)

        zi = self.projector(hi)
        zj = self.projector(hj)

        return hi, hj, zi, zj
    





class NT_XentLoss(nn.Module):
    def __init__(self, batch_size=512, temperature=0.5, **kwargs):
        super(NT_XentLoss, self).__init__()

        self.temperature = temperature

        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    
    def forward(self, zi, zj):
        N = zi.shape[0]

        z = torch.cat((zi, zj), dim=0)

        sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / self.temperature

        sim_matrix.fill_diagonal_(-1e9)

        labels = torch.cat((torch.arange(N) + N, torch.arange(N))).to(zi.device)

        loss = self.criterion(sim_matrix, labels)

        return loss / (2 * N)