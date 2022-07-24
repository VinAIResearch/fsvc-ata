import numpy as np
import torch


class ATA:
    def __init__(self, num_segments=8, sigma=1.0):
        # Temporal Order-Preserving prior
        T = (
            (torch.arange(num_segments).view(num_segments, 1) - torch.arange(num_segments).view(1, num_segments))
            / float(num_segments)
            / np.sqrt(2 / 64)
        )
        T = -(T**2) / 2.0 / (sigma**2)
        T = 1 / (sigma * np.sqrt(2 * np.pi)) * torch.exp(T)
        T = T / T.sum(1, keepdim=True)
        self.T = T.cuda()

    def similarity_matrix(self, query, support, scale, bias=None):
        """query : B x num_segments x C
        support : B x num_segments x C
        """
        D = query.unsqueeze(2).unsqueeze(2) * support.transpose(0, 1).unsqueeze(0).unsqueeze(0)
        if bias is not None:
            D = D.sum(-1) * scale + bias.transpose(0, 1).unsqueeze(0).unsqueeze(0)
        else:
            D = D.sum(-1) * scale
        normalized_D = D.softmax(2)
        return D, normalized_D

    def appearance_score(self, similarity_matrix, scale=5.0):
        sim_a = torch.logsumexp(similarity_matrix, 2).mean(1) * scale
        return sim_a

    def temporal_score(self, normalize_similarity_matrix, scale=5.0):
        kl = (
            (
                normalize_similarity_matrix
                * torch.log(normalize_similarity_matrix / self.T.unsqueeze(0).unsqueeze(-1) + 1e-12)
            )
            .sum(2)
            .mean(1)
        )
        sim_d = (2 - kl) * scale
        return sim_d
