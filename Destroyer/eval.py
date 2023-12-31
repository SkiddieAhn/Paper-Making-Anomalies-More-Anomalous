import sys, os
sys.path.append('.')
print('================================')
print(os.getcwd())
print('================================')

import torch 
import argparse
from evaluation.eval_log import * 
from evaluation.eval_train import val_train_eval
from evaluation.eval_test import val_test_eval
from evaluation.multigpu import MultiGPU

from config import update_config
from network.GUNet import UNet as GUNet
from network.DUNet import UNet as DUNet


parser = argparse.ArgumentParser(description='MAMA_Destroyer')
parser.add_argument('--dataset', default='avenue', type=str, help='The name of the dataset to evaluate.')
parser.add_argument('--trained_model', default=None, type=str, help='The pre-trained model to evaluate.')
parser.add_argument('--show_status', default=False, type=bool, help='show status')
parser.add_argument('--save_data', default=False, type=bool, help='save image, psnr')
parser.add_argument('--gaussian', default=False, type=bool, help='use gauissian 1d filter to Anomaly Score.')


def val(cfg, train_scores=None, models=None, iter=None, train=False):
    '''
    ========================================
    This is for evaluation during training.    
    ========================================
    '''
    if models:  
        train = True
        generator = models['generator']
        generator.eval()

        flownet = models['flownet']
        flownet.eval()

        segnet = models['segnet']
        segnet.eval()

        autoencoder = models['autoencoder']
        autoencoder.eval()

        g_apsnr, g_auc, a_auc, train_scores = val_train_eval(cfg, train_scores, generator, flownet, segnet, autoencoder, iter)

        return g_apsnr, g_auc, a_auc, train_scores
 
    '''
    ========================================
    This is for evaluation during testing.    
    ========================================
    '''
    if train == False:
        generator = GUNet(input_channels1=12, input_channels2=6, input_channels3=84, output_channels=3).cuda().eval()
        autoencoder = DUNet(3, 3).cuda().eval()

        # load weight
        generator.load_state_dict(torch.load('weights/' + cfg.trained_model + '.pth')['net_g'])
        autoencoder.load_state_dict(torch.load('weights/' + cfg.trained_model + '.pth')['net_a'])
        iter = torch.load('weights/' + cfg.trained_model + '.pth')['step']

        return val_test_eval(cfg, generator, autoencoder, iter)
    

if __name__ == '__main__':
    args = parser.parse_args()
    test_cfg = update_config(args, mode='test')
    test_cfg.print_cfg()
    val(test_cfg)
