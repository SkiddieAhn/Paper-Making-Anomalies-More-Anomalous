import torch 
from utils import *
import argparse
from training.losses import *
from network.GUNet import UNet as GUNet
from network.DUNet import UNet as DUNet
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
    generator = GUNet(input_channels1=12, input_channels2=6, input_channels3=84, output_channels=3).cuda()
    generator = generator.eval()

    # autoencoder (= destroyer)
    autoencoder = DUNet(3,3).cuda()
    autoencoder = autoencoder.train()

    # flownet
    flownet = FlowNet2SD().cuda()
        
    # deeplab v3
    with torch.no_grad():
        segnet = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True).cuda()

    return generator, autoencoder, flownet, segnet


def def_optim_sch(cfg, ae):
    optim_A = torch.optim.AdamW(ae.parameters(), lr=cfg.a_lr, weight_decay=cfg.l2)
    if cfg.sch:
        sch_A = torch.optim.lr_scheduler.PolynomialLR(optimizer=optim_A, total_iters=cfg.epoch_size)
    else:
        sch_A = None
    return optim_A, sch_A


def load_models(cfg, generator, autoencoder, flownet, segnet):
    if cfg.resume:
        generator.load_state_dict(torch.load(cfg.resume)['net_g'])
    else:
        generator.apply(proposed_weights_init)

    # Frozen Generator
    for params in generator.parameters():
        params.requires_grad = False

    # load autoencoder
    autoencoder.apply(weights_init_normal)

    print('\n===========================================================')
    print(f'Frozen Generator, AutoEncoder Ok!')
    print('===========================================================')

    # load flownet
    flownet.load_state_dict(torch.load('pretrained_flownet/FlowNet2-SD.pth')['state_dict'])
    flownet.eval()

    print('\n===========================================================')
    print(f'Pretrained FlowNet Ok!')
    print('===========================================================')

    # segmentation network mode change
    segnet.eval()

    print('\n===========================================================')
    print(f'Pretrained Segmentation Network Ok!')
    print('===========================================================')


def load_scores():
    '''
    g: generator 
    a: autoencoder
    f: final 
    '''
 
    step, iter_list = 0, []
    g_best_auc, a_best_auc = 0, 0
    g_auc_list, a_auc_list = [], []

    scores = dict()
    scores['step'] = step
    scores['iter_list'] = iter_list

    scores['g_best_auc'] = g_best_auc
    scores['a_best_auc'] = a_best_auc

    scores['g_auc_list'] = g_auc_list
    scores['a_auc_list'] = a_auc_list

    return scores


def make_model_dict(generator, autoencoder, flownet, segnet):
    models = dict()
    models['generator'] = generator
    models['autoencoder'] = autoencoder
    models['flownet'] = flownet
    models['segnet'] = segnet
    return models


def make_opt_dict(optimizer_A):
    opts = dict()
    opts['optimizer_A'] = optimizer_A
    return opts


def make_sch_dict(cfg, sch_A):
    schs = dict()
    if cfg.sch:
        schs['sch_A'] = sch_A
    else:
        schs['sch_A'] = None
    return schs