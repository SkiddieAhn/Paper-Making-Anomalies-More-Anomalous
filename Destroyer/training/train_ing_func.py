import torch 
import cv2
import numpy as np


def np_load_frame(filename, resize_h, resize_w):
    img = cv2.imread(filename)
    image_resized = cv2.resize(img, (resize_w, resize_h)).astype('float32')
    image_resized = (image_resized / 127.5) - 1.0 
    image_resized = np.transpose(image_resized, [2, 0, 1])  # to (C, H, W)
    image_resized = torch.from_numpy(image_resized)
    return image_resized


def make_models_dict(models, opts, scores):
    model_dict = {'net_g': models['generator'].state_dict(), 
                'net_a': models['autoencoder'].state_dict(), 'optimizer_a': opts['optimizer_A'].state_dict(),
                'step':int(scores['step']), 'iter_list':scores['iter_list'], 
                'g_best_auc':float(scores['g_best_auc']), 'a_best_auc':float(scores['a_best_auc']), 
                'g_auc_list':scores['g_auc_list'], 'a_auc_list':scores['a_auc_list']}
    return model_dict


def update_best_model(score_type, score, model_type, iteration, cfg, models, opts, scores):
    '''
    * score_type: g_best_auc, a_best_auc
    * model_type: g, a
    '''
    if score > scores[score_type]:
        scores[score_type] = score
        model_dict = make_models_dict(models, opts, scores)
        save_path = f'weights/{cfg.work_num}_{score_type}_{cfg.dataset}.pth'
        torch.save(model_dict, save_path)
        print('\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        print(f"[best {model_type} {score_type} model] update! at {iteration} iteration!! [{model_type} {score_type}: {scores[score_type]:.3f}]")
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')