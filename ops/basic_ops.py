import torch


class Identity(torch.nn.Module):
    def forward(self, input):
        return input
