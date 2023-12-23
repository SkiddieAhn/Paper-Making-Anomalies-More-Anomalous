import os
import copy
import torch 
from sklearn import metrics
from evaluation.eval_log import * 
from sklearn.cluster import KMeans

from config import update_config
from Dataset import Label_loader
import Dataset
from utils import *
import numpy as np
from network.non_generator.flownet2.models import FlowNet2SD
from einops import rearrange

def z_score(arr, eps=1e-8):
    mean = np.mean(arr)
    std_dev = np.std(arr) + eps   # Avoid division by zero
    z_scores = (arr - mean) / std_dev
    return z_scores

def min_max_normalize(arr, eps=1e-8):
    min_val = np.min(arr)
    max_val = np.max(arr)
    denominator = max_val - min_val + eps  # Avoid division by zero

    normalized_arr = (arr - min_val) / denominator
    return normalized_arr


def val_train_eval(cfg, train_scores, generator, flownet, segnet, iter):
    dataset_name = cfg.dataset
    video_folders = os.listdir(cfg.test_data)
    video_folders.sort()
    video_folders = [os.path.join(cfg.test_data, aa) for aa in video_folders]
    video_length = len(video_folders)

    g_sse_group = [] # generator sse group
    g_siml_m_group = [] # generator sim loss group
    g_siml_l_group = [] # generator sim loss group

    # return values
    g_auc = 0

    # Training Log
    if not os.path.exists(f"results/{dataset_name}"):
        os.makedirs(f"results/{dataset_name}")
    
    '''
    ===========================
    1. get PSNR Error 
    ===========================
    '''
    with torch.no_grad():
        # normal strategy
        for _, folder in enumerate(video_folders):
            one_video = Dataset.test_dataset(cfg, folder)

            g_video_sse = []  
            g_video_siml_m = []
            g_video_siml_l = []

            for _, clip in enumerate(one_video):
                # make frame input
                frame_1 = clip[0:3, :, :].unsqueeze(0).cuda()  # (1, 3, 256, 256) 
                frame_2 = clip[3:6, :, :].unsqueeze(0).cuda()  # (1, 3, 256, 256) 
                frame_3 = clip[6:9, :, :].unsqueeze(0).cuda()  # (1, 3, 256, 256) 
                frame_4 = clip[9:12, :, :].unsqueeze(0).cuda()  # (1, 3, 256, 256) 

                # make motion input
                flow_1_input = torch.cat([frame_1.unsqueeze(2), frame_2.unsqueeze(2)], 2).cuda() # (1, 3, 2, 256, 256)
                flow_1 = (flownet(flow_1_input * 255.) / 255.).detach().cuda() # (1, 2, 256, 256)
                flow_2_input = torch.cat([frame_2.unsqueeze(2), frame_3.unsqueeze(2)], 2).cuda() # (1, 3, 2, 256, 256)
                flow_2 = (flownet(flow_2_input * 255.) / 255.).detach().cuda() # (1, 2, 256, 256)
                flow_3_input = torch.cat([frame_3.unsqueeze(2), frame_4.unsqueeze(2)], 2).cuda() # (1, 3, 2, 256, 256)
                flow_3 = (flownet(flow_3_input * 255.) / 255.).detach().cuda() # (1, 2, 256, 256)

                # make label input
                seg_input = torch.cat([frame_1, frame_2, frame_3, frame_4], 0).cuda() # (4, 3, 256, 256)
                seg_output = segnet(seg_input)['out'].cuda() # (4, 21, 256, 256)
                seg_output = rearrange(seg_output, 'b c h w -> (b c) h w').unsqueeze(0).cuda() # (1, 84, 256, 256)
                label_input = seg_output
                
                # all input 
                input1 = torch.cat([frame_1,frame_2, frame_3, frame_4], 1).cuda() # (1, 12, 256, 256) 
                input2 = torch.cat([flow_1, flow_2, flow_3], 1).cuda() # (1, 6, 256, 256)
                input3 = label_input.cuda() # (1, 84, 256, 256)
                target_frame = clip[12:15, :, :].unsqueeze(0).cuda() # (1, 3, 256, 256)


                '''
                ---------------------------
                Future frame prediction
                ---------------------------
                '''
                G_frame, frame_feature, motion_feature, ftom_feature, label_feature, ftol_feature = generator(input1, input2, input3) # future frame prediction 

                g_test_sse = SSE(G_frame, target_frame).cpu().detach().numpy()
                g_video_sse.append(float(g_test_sse))
                g_test_siml_m = SIM_LOSS(anchor=ftom_feature, positive=motion_feature, negative=frame_feature).cpu().detach().numpy()
                g_video_siml_m.append(float(g_test_siml_m))
                g_test_siml_l = SIM_LOSS(anchor=ftol_feature, positive=label_feature, negative=frame_feature).cpu().detach().numpy()
                g_video_siml_l.append(float(g_test_siml_l))

                torch.cuda.synchronize()

            g_sse_group.append(np.array(g_video_sse))
            g_siml_m_group.append(np.array(g_video_siml_m))
            g_siml_l_group.append(np.array(g_video_siml_l))


    '''
    ================
    2. get Best AUC
    ================
    '''
    gt_loader = Label_loader(cfg, video_folders)  # Get gt labels.
    gt = gt_loader()
    assert len(g_sse_group) == len(gt), f'Ground truth has {len(gt)} videos, but got {len(g_sse_group)} detected videos.'


    for net, g_sse_group in enumerate([g_sse_group]):
        if net == 0:
            net = 'Generator'
        
        video_length = len(g_sse_group)

        best_auc = 0 
        best_weight = []

        for a in np.arange(0.1, 4.9, 0.1):
            a = round(a, 1)
            rest = round(5-a, 1)

            for b in np.arange(0.1, rest, 0.1):
                b = round(b, 1)
                c = round(rest-b, 1)

                # init
                scores = np.array([], dtype=np.float32)
                labels = np.array([], dtype=np.int8)

                for i in range(video_length):
                    # anomaly score 
                    siml_m = np.copy(g_siml_m_group[i])       
                    siml_m = z_score(siml_m)   

                    siml_l = np.copy(g_siml_l_group[i])       
                    siml_l = z_score(siml_l)   

                    sse = np.copy(g_sse_group[i]) 
                    sse = z_score(sse)   

                    distance = (a*siml_m)+(b*siml_l)+(c*sse)
                    distance = min_max_normalize(distance)

                    label = gt[i][4:]
                    scores = np.concatenate((scores, distance), axis=0)
                    labels = np.concatenate((labels, label), axis=0)  

                fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
                auc = metrics.auc(fpr, tpr)  

                # best model
                if auc > best_auc:
                    best_auc = auc    
                    best_weight = [a, b, c]


        '''
        ===========================
        3. Logging PSNR & AUC
        ===========================
        '''

        # etc
        if net == 'Generator':
            train_scores['iter_list'].append(iter) # append once only
            train_scores['g_auc_list'].append(best_auc)
            g_auc = best_auc 

            # AUC
            save_text(f"[{net}][{iter}] AUC: {best_auc}, weight: {best_weight}", f'results/{dataset_name}/psnrs-auc.txt')
            save_auc_graph_train(iters=train_scores['iter_list'], scores=train_scores['g_auc_list'], file_path=f'results/{dataset_name}/g_auc_itr_graph.jpg')

    return -1, g_auc, train_scores