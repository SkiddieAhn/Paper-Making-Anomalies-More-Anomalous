import numpy as np
import torch 
import torchvision.transforms as transforms
from einops import rearrange
from utils import *

class PatchGenerator:
    def __init__(self, img_size, patch_size):
        self.img_size = img_size
        self.patch_size = patch_size

    def __call__(self, img):
        '''
        if) patch_size = 8 
        [3x256x256] => [1024x3x8x8]
        '''
        patches = rearrange(img, "b c (h p1) (w p2) -> b (h w) c p1 p2", h=self.img_size//self.patch_size, w=self.img_size//self.patch_size, p1=self.patch_size, p2=self.patch_size)
        return patches

    def reverse(self, patches):
        '''
        if) patch_size = 8 
        [1024x3x8x8] => [3x256x256] 
        '''
        img = rearrange(patches,"b (h w) c p1 p2 -> b c (h p1) (w p2)", h=self.img_size//self.patch_size, w=self.img_size//self.patch_size, p1=self.patch_size, p2=self.patch_size)
        return img


def tensor_gaussian_blur(img, kernel_size, sigma):
    transform = transforms.GaussianBlur(kernel_size, sigma)
    blurred_img = transform(img) 

    return blurred_img


def sp_noise_gpu(image, prob):
    # Add salt and pepper noise to image (GPU implementation)
    output = image.clone()

    batch_size, num_channels, height, width = output.size()

    # Generate random noise mask
    noise_mask = torch.rand(batch_size, num_channels, height, width, device=output.device)

    # Create black and white pixels
    black = torch.zeros_like(output, device=output.device)
    white = torch.ones_like(output, device=output.device)

    # Apply salt and pepper noise
    output[noise_mask < (prob / 2)] = black[noise_mask < (prob / 2)]
    output[noise_mask > 1 - (prob / 2)] = white[noise_mask > 1 - (prob / 2)]

    return output


def make_noise_img(img, num, dropout):
        # Weak Blur Noise
        if num == 0:
            return tensor_gaussian_blur(img, 5, 2)

        # Strong Blur Noise
        elif num == 1:
            return tensor_gaussian_blur(img, 9, 2)

        # Dropout Noise (channel independent)
        elif num ==2:
            dropout = torch.nn.Dropout2d(p=dropout)
            output = dropout(img)
            return output
        
        # My Dropout Noise (channel dependent)
        elif num == 2.5: 
            dropout_mask = torch.rand_like(img[:, :1, :, :])
            output = img * (dropout_mask > dropout).float()
            return output

        # salt and pepper Noise
        elif num == 3:
            return sp_noise_gpu(img, 0.05)

        else:
            NotImplementedError

        return output


def noiseBbox(img, all=False, divide=2, method=0, dropout=0.3, num=1):
    '''
    type of img is torch.Tensor
    '''
    img_size = (img.shape[-2], img.shape[-1]) # (h, w)

    if all:
        noise_area_size = (img_size[0], img_size[1])
    else:
        noise_area_size = (img_size[0]//divide, img_size[1]//divide)

    positions = []

    for _ in range(num):
        # Noise position
        left = torch.randint(0, img_size[0] - noise_area_size[0] + 1, (1,))
        top = torch.randint(0, img_size[1] - noise_area_size[1] + 1, (1,))
        right = left + noise_area_size[0]
        bottom = top + noise_area_size[1]
        position = (left, top, right, bottom)

        # make noise img
        # 0: weak blur, 1: strong blur, 2: dropout, 3: salt and pepper
        selected = img[:, :, top.item():bottom.item(), left.item():right.item()]
        noisy_image = make_noise_img(selected, method, dropout)

        # add noise to Noise position
        img[:, :, top.item():bottom.item(), left.item():right.item()] = noisy_image

        positions.append(position)

    return img, positions


def randomNoiseBbox(img, num=1):
    '''
    type of img is torch.Tensor
    '''
    
    # random divide, method, dp_rate
    divide = int(torch.randint(2, 7, (1,))) # 2 ~ 6
    method = int(torch.randint(0, 3, (1,))) # 0 ~ 2
    dropout = round(0.05 + 0.25 * torch.rand(1).item(), 2) # 0.05 ~ 0.3

    # set bbox position 
    img_size = (img.shape[-2], img.shape[-1]) # (h, w)
    noise_area_size = (img_size[0]//divide, img_size[1]//divide)
    positions = []

    for _ in range(num):
        # Noise position
        left = torch.randint(0, img_size[0] - noise_area_size[0] + 1, (1,))
        top = torch.randint(0, img_size[1] - noise_area_size[1] + 1, (1,))
        right = left + noise_area_size[0]
        bottom = top + noise_area_size[1]
        position = (left, top, right, bottom)

        # make noise img
        # 0: weak blur, 1: strong blur, 2: dropout, 3: salt and pepper
        selected = img[:, :, top.item():bottom.item(), left.item():right.item()]
        noisy_image = make_noise_img(selected, method, dropout)

        # add noise to Noise position
        img[:, :, top.item():bottom.item(), left.item():right.item()] = noisy_image

        positions.append(position)

    return img, positions


def answerBbox(img, bg, positions):
    '''
    type of img is torch.Tensor
    '''
    for i in range(len(positions)):
        position = positions[i]

        # position
        left, top, right, bottom = position[0], position[1], position[2], position[3]

        # add bg to img
        selected = bg[:, top.item():bottom.item(), left.item():right.item()]
        b, _, _, _ = img.shape
        selected_expanded = selected.unsqueeze(0).repeat(b, 1, 1, 1)
        img[:, :, top.item():bottom.item(), left.item():right.item()] = selected_expanded

    return img


def noisePatches(pg, input, method=0, rate=0.3, dropout=0.3):
    # make patches 
    patches = pg(input).cuda()  # [batch, patch_num, channels, patch_size, patch_size]
    batch_size, patch_num, _, _, _ = patches.shape

    # make noise patches and record noise positions
    num = torch.rand(patch_num).cuda()
    mask = num < rate  
    mask = mask.view(1, patch_num, 1, 1, 1).repeat(batch_size, 1, 1, 1, 1) # reshape for broadcasting [b, pn, 1, 1, 1]

    # make noise patches
    b, p, c, ph, pw = patches.shape
    patches = rearrange(patches, "b p c ph pw -> (b p) c ph pw", b=b, p=p, c=c, ph=ph, pw=pw)
    noise_patches = make_noise_img(patches, method, dropout).cuda()
    noise_patches = rearrange(noise_patches, "(b p) c ph pw -> b p c ph pw", b=b, p=p, c=c, ph=ph, pw=pw)
    patches = rearrange(patches, "(b p) c ph pw -> b p c ph pw", b=b, p=p, c=c, ph=ph, pw=pw)

    # Apply the mask to update the patches
    patches = torch.where(mask, noise_patches, patches)

    # Record the noise positions
    mask = mask.squeeze(2).squeeze(2).squeeze(2) # [b, pn]
    mask = mask[0].int() # [pn]

    # make noise image
    image = pg.reverse(patches).cuda()

    return image, mask


def randomNoisePatches(pg, input, method=None):
    # make patches 
    patches = pg(input).cuda()  # [batch, patch_num, channels, patch_size, patch_size]
    _, patch_num, _, _, _ = patches.shape

    # random rate, method, dp_rate
    rate = round(0.05 + 0.45 * torch.rand(1).item(), 2) # 0.05 ~ 0.5
    dropout = round(0.05 + 0.45 * torch.rand(1).item(), 2) # 0.05 ~ 0.5
    if method == None:
        method = int(torch.randint(0, 3, (1,))) # 0 ~ 2

    # make noise patches and record noise positions
    num = torch.rand(patch_num).cuda()  
    mask = num < rate  
    mask = mask.view(1, patch_num, 1, 1, 1).repeat(input.shape[0], 1, 1, 1, 1) # reshape for broadcasting

    # make noise patches
    b, p, c, ph, pw = patches.shape
    patches = rearrange(patches, "b p c ph pw -> (b p) c ph pw", b=b, p=p, c=c, ph=ph, pw=pw)
    noise_patches = make_noise_img(patches, method, dropout).cuda()
    noise_patches = rearrange(noise_patches, "(b p) c ph pw -> b p c ph pw", b=b, p=p, c=c, ph=ph, pw=pw)
    patches = rearrange(patches, "(b p) c ph pw -> b p c ph pw", b=b, p=p, c=c, ph=ph, pw=pw)

    # Apply the mask to update the patches
    patches = torch.where(mask, noise_patches, patches)

    # Record the noise positions
    mask = mask.squeeze(2).squeeze(2).squeeze(2)
    mask = mask[0].int()

    # make noise image
    image = pg.reverse(patches).cuda()

    return image, mask


def answerPatches(pg, input, bg, mask):
    # set bg
    bg = bg.unsqueeze(0).repeat(input.shape[0], 1, 1, 1) # [b, c, h, w]

    # make patches
    patches_in = pg(input).cuda() # [batch, patch_num, channels, patch_size, patch_size]
    patches_bg = pg(bg).cuda()

    # make answer patches
    batch_size, patch_num, _, _, _ = patches_in.shape
    mask = mask.bool()
    mask = mask.view(1, patch_num, 1, 1, 1).repeat(batch_size, 1, 1, 1, 1) # reshape for broadcasting [b, pn, 1, 1, 1]
    patches = torch.where(mask, patches_bg, patches_in)

    # make answer image
    image = pg.reverse(patches).cuda()

    return image