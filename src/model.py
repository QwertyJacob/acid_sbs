import torch
import torch.nn as nn


class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()
        layers = []
        for i in range(n_layers):
            layers += [
                nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # [B, hidden_dim]


class PrototypicalClassifier(nn.Module):
    """Stateless: prototypes are computed on every forward pass from the support set."""

    def forward(
        self,
        support_embeddings: torch.Tensor,  # [N_classes * k_shot, hidden_dim]
        support_labels: torch.Tensor,      # [N_classes * k_shot]
        query_embeddings: torch.Tensor,    # [N_classes * n_query, hidden_dim]
    ):
        """
        Returns:
            logits:     [n_query_total, n_classes]   negative squared distances
            prototypes: [n_classes, hidden_dim]
        """
        unique_labels = support_labels.unique(sorted=True)
        prototypes = torch.stack([
            support_embeddings[support_labels == c].mean(dim=0)
            for c in unique_labels
        ])  # [n_classes, hidden_dim]

        # dists[i, j] = || query_i - proto_j ||^2
        dists = torch.cdist(query_embeddings, prototypes).pow(2)  # [n_query, n_classes]
        logits = -dists  # higher = closer = more likely
        return logits, prototypes
