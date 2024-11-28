import torch
import torch.nn as nn
import time
import os
import numpy as np

from rectified_flow.flow_components.interpolation_solver import AffineInterp

class AnalyticGaussianVelocity(nn.Module):
    def __init__(
        self, 
        dataset: torch.Tensor,
        interp: AffineInterp,
    ):
        super().__init__()
        self.dataset = nn.Parameter(dataset.detach().clone())  # (N_data, D)
        self.dataset.requires_grad = False
        self.dataset_norm = self.dataset.norm(dim=1).pow(2)
        self.interp = interp

    def forward(self, x_t, t):
        at, bt, dot_at, dot_bt = self.interp.get_coeffs(t, detach=True)
        return self.get_analytic_velocity(x_t, at, dot_at, bt, dot_bt)

    @torch.no_grad()
    def get_analytic_velocity(self, x_t, a_t, dot_a_t, b_t, dot_b_t):
        """
        x_t shape: (Batch, D)
        a_t, dot_a_t, b_t, dot_b_t shape: (Batch,)
        """
        term_1 = x_t.norm(dim=1).pow(2)
        term_2 = torch.einsum("bd, nd -> bn", [x_t, self.dataset])
        term_3 = self.dataset_norm
        
        logit = term_1[:, None] - 2. * a_t[:, None] * term_2 + a_t.pow(2)[:, None] * term_3[None, :]
        logit = ((-1.) / (2. * b_t.pow(2)))[:, None] * logit
        
        prob = torch.softmax(logit, dim=1)         # (Batch, N_data)
        data_coeff = dot_a_t - a_t * dot_b_t / b_t # (Batch,)
        prob = prob * data_coeff[:, None]
        weighted_dataset = torch.einsum("bn, nd -> bd", [prob, self.dataset])
        
        v = (dot_b_t / b_t)[:, None] * x_t + weighted_dataset

        # print(f"term_1: {term_.shape}")      # (Batch,)
        # print(f"term_2: {term_2.shape}")     # (Batch, N_data)
        # print(f"term_3: {term_3.shape}")     # (N_data,)
        # print(f"logit: {logit.shape}")       # (Batch, N_data)
        # print(v.shape)
        # print(f"weighted_dataset: {weighted_dataset.shape}")  # weighted_dataset: (Batch, D)

        return v
