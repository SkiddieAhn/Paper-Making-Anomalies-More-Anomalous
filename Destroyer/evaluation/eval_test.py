import os, time
import copy
import torch 
from sklearn import metrics
from evaluation.eval_log import * 
from scipy.ndimage import gaussian_filter1d

from config import update_config
from Dataset import Label_loader
import Dataset
from utils import *
from torchvision.utils import save_image
import numpy as np
from network.non_generator.flownet2.models import FlowNet2SD
from einops import rearrange


def calculate_eer(fpr, tpr):
    min_diff = float('inf')
    eer = 0.0

    for i in range(len(fpr)):
        diff = abs(fpr[i] - (1 - tpr[i]))
        if diff < min_diff:
            min_diff = diff
            eer = fpr[i]
    return eer


def z_score(arr, eps=1e-8):
    mean = np.mean(arr)
    std_dev = np.std(arr) + eps  # Avoid division by zero
    z_scores = (arr - mean) / std_dev
    return z_scores


def min_max_normalize(arr, eps=1e-8):
    min_val = np.min(arr)
    max_val = np.max(arr)
    denominator = max_val - min_val + eps  # Avoid division by zero

    normalized_arr = (arr - min_val) / denominator
    return normalized_arr


def val_test_eval(cfg, generator, autoencoder, iter):
    dataset_name = cfg.dataset
    video_folders = os.listdir(cfg.test_data)
    video_folders.sort()
    video_folders = [os.path.join(cfg.test_data, aa) for aa in video_folders]
    video_length = len(video_folders)

    g_sse_group = [] # generator sse group
    a_sse_group = []
    g_siml_m_group = [] # generator sim loss group
    g_siml_l_group = [] # generator sim loss group

    fps = 0 

    flownet = FlowNet2SD().cuda()
    flownet.load_state_dict(torch.load('pretrained_flownet/FlowNet2-SD.pth')['state_dict'])
    flownet.eval()

    with torch.no_grad():
        segnet = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True).cuda()
    segnet.eval()

    if not os.path.exists(f"results/{dataset_name}/{iter}"):
        os.makedirs(f"results/{dataset_name}/{iter}")

    '''
    ===========================
    1. get PSNR Error 
    ===========================
    '''
    with torch.no_grad():
        for i, folder in enumerate(video_folders):
            if cfg.save_data:
                # Testing Log
                if not os.path.exists(f"results/{dataset_name}/{iter}/f{i+1}/generator"):
                    os.makedirs(f"results/{dataset_name}/{iter}/f{i+1}/generator")
                if not os.path.exists(f"results/{dataset_name}/{iter}/f{i+1}/autoencoder"):
                    os.makedirs(f"results/{dataset_name}/{iter}/f{i+1}/autoencoder")
                if not os.path.exists(f"results/{dataset_name}/{iter}/f{i+1}/target"):
                    os.makedirs(f"results/{dataset_name}/{iter}/f{i+1}/target")

            one_video = Dataset.test_dataset(cfg, folder)

            g_video_sse = []  
            a_video_sse = []
            g_video_siml_m = []
            g_video_siml_l = []

            for j, clip in enumerate(one_video):
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
                --------------------------------------------
                Calc PSNR with Prediction & Denoising Output
                --------------------------------------------
                '''
                G_frame, frame_feature, motion_feature, ftom_feature, label_feature, ftol_feature = generator(input1, input2, input3) # future frame prediction 
                A_frame = autoencoder(G_frame)

                a_test_sse = SSE(A_frame, target_frame).cpu().detach().numpy()
                a_video_sse.append(float(a_test_sse))
                g_test_sse = SSE(G_frame, target_frame).cpu().detach().numpy()
                g_video_sse.append(float(g_test_sse))
                g_test_siml_m = SIM_LOSS(anchor=ftom_feature, positive=motion_feature, negative=frame_feature).cpu().detach().numpy()
                g_video_siml_m.append(float(g_test_siml_m))
                g_test_siml_l = SIM_LOSS(anchor=ftol_feature, positive=label_feature, negative=frame_feature).cpu().detach().numpy()
                g_video_siml_l.append(float(g_test_siml_l))

                if cfg.save_data:
                    g_test_psnr = psnr_error(G_frame, target_frame).cpu().detach().numpy()
                    a_test_psnr = psnr_error(A_frame, target_frame).cpu().detach().numpy()

                    g_res_temp = ((G_frame[0] + 1 ) / 2)[(2,1,0),...]
                    a_res_temp = ((A_frame[0] + 1 ) / 2)[(2,1,0),...]
                    t_res_temp = ((target_frame[0] + 1 ) / 2)[(2,1,0),...]

                    save_image(g_res_temp, f'results/{dataset_name}/{iter}/f{i+1}/generator/{j}_img.jpg')
                    save_image(a_res_temp, f'results/{dataset_name}/{iter}/f{i+1}/autoencoder/{j}_img.jpg')
                    save_image(t_res_temp, f'results/{dataset_name}/{iter}/f{i+1}/target/{j}_img.jpg')
                    save_text(f"[Generator] {j}: {g_test_psnr} psnr", f'results/{dataset_name}/{iter}/f{i+1}/generator/psnrs.txt')
                    save_text(f"[AutoEncoder] {j}: {a_test_psnr} psnr", f'results/{dataset_name}/{iter}/f{i+1}/autoencoder/psnrs.txt')
                
                torch.cuda.synchronize()
                end = time.time()
                if j > 1:  # Compute fps by calculating the time used in one completed iteration, this is more accurate.
                    fps = 1 / (end - temp)
                temp = end

                if cfg.show_status:
                    print(f'\rDetecting: [{i + 1:02d}] {j + 1}/{len(one_video)}, {fps:.2f} fps.', end='')

            g_sse_group.append(np.array(g_video_sse))
            a_sse_group.append(np.array(a_video_sse))
            g_siml_m_group.append(np.array(g_video_siml_m))
            g_siml_l_group.append(np.array(g_video_siml_l))

        if cfg.show_status:
            print('\nAll frames were detected, begin to compute Regular Score and AUC.')

    gt_loader = Label_loader(cfg, video_folders)  # Get gt labels.
    gt = gt_loader()

    assert len(g_sse_group) == len(gt), f'Ground truth has {len(gt)} videos, but got {len(g_sse_group)} detected videos.'

    '''
    ================================================
    2-1. get AUC Error (generator, destroyer score)
    ================================================
    '''

    for net, sse_group in enumerate([g_sse_group, a_sse_group]):
        if net == 0:
            net = 'Generator'
        else:
            net = 'Autoencoder'
        
        video_length = len(sse_group)

        best_fpr = []
        best_tpr = []
        best_weight = []
        best_auc = 0
        best_eer = 0 

        for a in np.arange(0.1, 4.9, 0.1):
            a = round(a, 1)
            rest = round(5-a, 1)

            for b in np.arange(0.1, rest, 0.1):
                b = round(b, 1)
                c = round(rest-b, 1)

                for d in [0.25, 0.5, 0.75, 1.0]:
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

                        if net == 'Generator':
                            distance = (a*siml_m)+(b*siml_l)+(c*sse)
                        else:
                            sse2 = np.copy(a_sse_group[i])
                            sse2 = z_score(sse2)
                            distance = (a*siml_m)+(b*siml_l)+(c*sse)+(d*sse2)
                            
                        distance = min_max_normalize(distance)
                        # use gaussian 1d filter to Anomaly Score
                        if cfg.gaussian:
                            distance = gaussian_filter1d(distance, sigma=10)

                        label = gt[i][4:]
                        scores = np.concatenate((scores, distance), axis=0)
                        labels = np.concatenate((labels, label), axis=0)  

                    fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
                    auc = metrics.auc(fpr, tpr)  

                    # best model
                    if auc > best_auc:
                        best_auc = auc
                        best_fpr = fpr
                        best_tpr = tpr    
                        best_eer = calculate_eer(fpr=best_fpr, tpr=best_tpr)
                        if net == 'Generator':    
                            best_weight = [a, b, c]
                            break
                        else:
                            best_weight = [a, b, c, d]

        # Report AUC
        if net == 'Generator':
            save_auc_graph_test(best_fpr, best_tpr, best_auc, eer=best_eer, file_path=f'results/{dataset_name}/{iter}/g_total_auc_curve.jpg')
            save_text(f"generator auc/eer: {best_auc} auc, {best_eer} eer, weight: {best_weight}\n\n", f'results/{dataset_name}/{iter}/g_auc.txt')
            print(f'generator auc/eer: {best_auc} auc, {best_eer} eer\n')
        else:
            save_auc_graph_test(best_fpr, best_tpr, best_auc, eer=best_eer, file_path=f'results/{dataset_name}/{iter}/a_total_auc_curve.jpg')
            save_text(f"autoencoder auc/eer: {best_auc} auc, {best_eer} eer, weight: {best_weight}\n\n", f'results/{dataset_name}/{iter}/a_auc.txt')
            print(f'autoencoder auc/eer: {best_auc} auc, {best_eer} eer\n')


