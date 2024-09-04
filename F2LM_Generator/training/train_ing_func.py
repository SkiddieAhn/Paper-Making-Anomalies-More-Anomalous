import torch 

def make_models_dict(models, opts, scores):
    model_dict = {'net_g': models['generator'].state_dict(), 'optimizer_g': opts['optimizer_G'].state_dict(),
                'net_d': models['discriminator'].state_dict(), 'optimizer_d': opts['optimizer_D'].state_dict(),
                'step':int(scores['step']), 'iter_list':scores['iter_list'], 
                'g_best_auc':float(scores['g_best_auc']), 'g_auc_list':scores['g_auc_list']}
    return model_dict


def update_best_model(score_type, score, model_type, iteration, cfg, models, opts, scores):
    '''
    * score_type: g_best_auc
    * model_type: g
    '''
    if score > scores[score_type]:
        scores[score_type] = score
        model_dict = make_models_dict(models, opts, scores)
        save_path = f'weights/{cfg.work_num}_{score_type}_{cfg.dataset}.pth'
        torch.save(model_dict, save_path)
        print('\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        print(f"[best {model_type} {score_type} model] update! at {iteration} iteration!! [{model_type} {score_type}: {scores[score_type]:.3f}]")
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')