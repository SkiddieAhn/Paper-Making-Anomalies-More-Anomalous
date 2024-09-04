import torch
import torch.nn as nn
import torch.nn.functional
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from training.ssim_set import ssim_error
from torch.nn.modules.distance import PairwiseDistance


class getDestroyerLoss(nn.Module):
    def __init__(self, pg, ld=1):
        super().__init__()
        self.pg = pg
        self.ld = ld

    def patch_destroyer_loss(self, A, N, GT, G, BG):
        batch_size, patch_num = A.shape[0], A.shape[1]
        ssim_scores = torch.zeros((batch_size, patch_num)).cuda()  # [batch, patch_num]

        for b in range(batch_size):
            for p in range(patch_num):
                n_patch = N[b, p].unsqueeze(0)  # [1, patch_channel, patch_height, patch_width]
                gt_patch = GT[b, p].unsqueeze(0)
                ssim_scores[b, p] = ssim_error(img1=n_patch, img2=gt_patch)

        diff = torch.min(self.ld * (1-ssim_scores), torch.ones_like(ssim_scores)) # [batch, patch_num]

        bg_loss = torch.mean(torch.pow(BG - G, 2), dim=(2, 3, 4))
        a_loss = torch.mean(torch.pow(A - G, 2), dim=(2, 3, 4))

        loss = diff * bg_loss + (1 - diff) * a_loss
        return torch.sum(loss)

    def forward(self, A, N, GT, G, BG):
        '''
        A: Generator output,
        N: Generator output + Noise (Destroyer input),
        GT: Generator target,
        G: Destroyer output,
        BG: background image
        [shape: (batch, channel, height, width)]
        '''
        # make patches
        patches_A = self.pg(A) # [batch, patch_num, channel, patch_height, patch_width]
        patches_N = self.pg(N)
        patches_GT = self.pg(GT)
        patches_G = self.pg(G)
        patches_BG = self.pg(BG)

        # get loss
        output = self.patch_destroyer_loss(A=patches_A, N=patches_N, GT=patches_GT, G=patches_G, BG=patches_BG)
        return output
    