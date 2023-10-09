import torch
import torch.nn as nn
import torch.nn.functional
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from training.ssim_set import ssim_error
from torch.nn.modules.distance import PairwiseDistance


class cosine_similarity_loss(torch.nn.Module):
    def __init__(self, margin=0.2):
        super(cosine_similarity_loss, self).__init__()
        self.margin = margin
        self.pdist = PairwiseDistance(2)

    def forward(self, anchor, positive, negative):
        pos_dist = self.pdist.forward(anchor, positive)
        neg_dist = self.pdist.forward(anchor, negative)

        hinge_dist = torch.clamp(self.margin + pos_dist - neg_dist, min=0.0)
        loss = torch.mean(hinge_dist)
        return loss
    

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
    

class Flow_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gen_flows, gt_flows):
        return torch.mean(torch.abs(gen_flows - gt_flows))

class Intensity_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gen_frames, gt_frames):
        return torch.mean(torch.abs((gen_frames - gt_frames) ** 2))


class Gradient_Loss(nn.Module):
    def __init__(self, channels):
        super().__init__()

        pos = torch.from_numpy(np.identity(channels, dtype=np.float32))
        neg = -1 * pos
        # Note: when doing conv2d, the channel order is different from tensorflow, so do permutation.
        self.filter_x = torch.stack((neg, pos)).unsqueeze(0).permute(3, 2, 0, 1).cuda()
        self.filter_y = torch.stack((pos.unsqueeze(0), neg.unsqueeze(0))).permute(3, 2, 0, 1).cuda()

    def forward(self, gen_frames, gt_frames):
        # Do padding to match the  result of the original tensorflow implementation
        gen_frames_x = nn.functional.pad(gen_frames, [0, 1, 0, 0])
        gen_frames_y = nn.functional.pad(gen_frames, [0, 0, 0, 1])
        gt_frames_x = nn.functional.pad(gt_frames, [0, 1, 0, 0])
        gt_frames_y = nn.functional.pad(gt_frames, [0, 0, 0, 1])

        gen_dx = torch.abs(nn.functional.conv2d(gen_frames_x, self.filter_x))
        gen_dy = torch.abs(nn.functional.conv2d(gen_frames_y, self.filter_y))
        gt_dx = torch.abs(nn.functional.conv2d(gt_frames_x, self.filter_x))
        gt_dy = torch.abs(nn.functional.conv2d(gt_frames_y, self.filter_y))

        grad_diff_x = torch.abs(gt_dx - gen_dx)
        grad_diff_y = torch.abs(gt_dy - gen_dy)

        return torch.mean(grad_diff_x + grad_diff_y)


class Adversarial_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, fake_outputs):
        # TODO: compare with torch.nn.MSELoss ?
        return torch.mean((fake_outputs - 1) ** 2 / 2)


class Discriminate_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, real_outputs, fake_outputs):
        return torch.mean((real_outputs - 1) ** 2 / 2) + torch.mean(fake_outputs ** 2 / 2)


# if __name__ == '__main__':
#     # Debug Gradient_Loss, mainly on the padding issue.
#     import numpy as np
#
#     aa = torch.tensor([[1, 2, 3, 4, 2],
#                        [11, 12, 13, 14, 12],
#                        [1, 2, 3, 4, 2],
#                        [21, 22, 23, 24, 22],
#                        [1, 2, 3, 4, 2]], dtype=torch.float32)
#
#     aa = aa.repeat(4, 3, 1, 1)
#
#     pos = torch.from_numpy(np.identity(3, dtype=np.float32))
#     neg = -1 * pos
#     filter_x = torch.stack((neg, pos)).unsqueeze(0).permute(3, 2, 0, 1)
#     filter_y = torch.stack((pos.unsqueeze(0), neg.unsqueeze(0))).permute(3, 2, 0, 1)
#
#     gen_frames_x = nn.functional.pad(aa, [0, 1, 0, 0])
#     gen_frames_y = nn.functional.pad(aa, [0, 0, 0, 1])
#
#     gen_dx = torch.abs(nn.functional.conv2d(gen_frames_x, filter_x))
#     gen_dy = torch.abs(nn.functional.conv2d(gen_frames_y, filter_y))
#
#
#     print(aa)
#     print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
#     print(filter_y)  # (2, 1, 3, 3)
#     print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
#     print(gen_dx)
