import torch 
from utils import *
import argparse
from training.losses import *
from training.multigpu import MultiGPU
from network.unet import UNet
from network.non_generator.pix2pix_networks import PixelDiscriminator
from network.non_generator.flownet2.models import FlowNet2SD


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

def print_infor(cfg, dataloader):
    cfg.epoch_size = cfg.iters // len(dataloader)
    cfg.print_cfg() 

    print('\n===========================================================')
    print('Dataloader Ok!')
    print('-----------------------------------------------------------')
    print('[Data Size]:',len(dataloader.dataset))
    print('[Batch Size]:',cfg.batch_size)
    print('[One epoch]:',len(dataloader),'step   # (Data Size / Batch Size)')
    print('[Epoch & Iteration]:',cfg.epoch_size,'epoch &', cfg.iters,'step')
    print('-----------------------------------------------------------')
    print('===========================================================')


def seed(seed_value):
    if seed_value == -1:
        return

    # Otherwise seed all functionality
    import random
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True


def def_models():
    # generator
    generator = UNet(mode='train', input_channels1=12, input_channels2=6, input_channels3=84, output_channels=3).cuda()
    generator = generator.train()
    
    # discriminator
    discriminator = PixelDiscriminator(input_nc=3).cuda()
    discriminator = discriminator.train()

    # flownet
    flownet = FlowNet2SD().cuda()
        
    # deeplab v3
    with torch.no_grad():
        segnet = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True).cuda()

    return generator, discriminator, flownet, segnet


def def_losses():
    adversarial_loss = Adversarial_Loss().cuda()
    discriminate_loss = Discriminate_Loss().cuda()
    gradient_loss = Gradient_Loss(3).cuda()
    intensity_loss = Intensity_Loss().cuda()
    triplet_loss = Triplet_loss().cuda()

    return adversarial_loss, discriminate_loss, gradient_loss, intensity_loss, triplet_loss


def def_optim_sch(cfg, gen, disc):
    optim_G = torch.optim.AdamW(gen.parameters(), lr=cfg.g_lr, weight_decay=cfg.l2)
    optim_D = torch.optim.AdamW(disc.parameters(), lr=cfg.d_lr, weight_decay=cfg.l2)
    if cfg.sch:
        sch_G = torch.optim.lr_scheduler.PolynomialLR(optimizer=optim_G, total_iters=cfg.epoch_size)
        sch_D = torch.optim.lr_scheduler.PolynomialLR(optimizer=optim_D, total_iters=cfg.epoch_size)
    else:
        sch_G = None
        sch_D = None
    return optim_G, optim_D, sch_G, sch_D


def load_models(cfg, generator, discriminator, flownet, segnet, optimizer_G, optimizer_D):
    if cfg.resume:
        generator.load_state_dict(torch.load(cfg.resume)['net_g'])
        discriminator.load_state_dict(torch.load(cfg.resume)['net_d'])
        optimizer_G.load_state_dict(torch.load(cfg.resume)['optimizer_g'])
        optimizer_D.load_state_dict(torch.load(cfg.resume)['optimizer_d'])
    else:
        generator.apply(proposed_weights_init)
        discriminator.apply(weights_init_normal)

    print('\n===========================================================')
    print(f'Generator, Discriminator Ok!')
    print('===========================================================')

    # load flownet
    flownet.load_state_dict(torch.load('pretrained_flownet/FlowNet2-SD.pth')['state_dict'])
    flownet.eval()

    print('\n===========================================================')
    print(f'Pretrained FlowNet Ok!')
    print('===========================================================')

    # change segnet mode
    segnet.eval()

    print('\n===========================================================')
    print(f'Pretrained Segmentation Network Ok!')
    print('===========================================================')


def load_scores(cfg):
    '''
    g: generator 
    a: autoencoder
    f: final 
    '''
    if cfg.resume:
        step = torch.load(cfg.resume)['step']
        iter_list = torch.load(cfg.resume)['iter_list']
        g_best_auc = torch.load(cfg.resume)['g_best_auc']
        g_auc_list = torch.load(cfg.resume)['g_auc_list']

    else:
        step, iter_list = 0, []
        g_best_auc = 0
        g_auc_list = []

    scores = dict()
    scores['step'] = step
    scores['iter_list'] = iter_list
    scores['g_best_auc'] = g_best_auc
    scores['g_auc_list'] = g_auc_list

    return scores


def make_model_dict(generator, discriminator, flownet, segnet):
    models = dict()
    models['generator'] = generator
    models['discriminator'] = discriminator
    models['flownet'] = flownet
    models['segnet'] = segnet
    return models
    

def make_loss_dict(discriminate_loss, intensity_loss, gradient_loss, adversarial_loss, triplet_loss):
    losses = dict()
    losses['discriminate_loss'] = discriminate_loss
    losses['intensity_loss'] = intensity_loss
    losses['gradient_loss'] = gradient_loss
    losses['adversarial_loss'] = adversarial_loss
    losses['triplet_loss'] = triplet_loss
    return losses


def make_opt_dict(optimizer_G, optimizer_D):
    opts = dict()
    opts['optimizer_G'] = optimizer_G
    opts['optimizer_D'] = optimizer_D
    return opts


def make_sch_dict(cfg, sch_G, sch_D):
    schs = dict()
    if cfg.sch:
        schs['sch_G'] = sch_G
        schs['sch_D'] = sch_D
    else:
        schs['sch_G'] = None
        schs['sch_D'] = None
    return schs