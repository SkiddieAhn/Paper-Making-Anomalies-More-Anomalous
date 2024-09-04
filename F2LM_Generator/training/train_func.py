import os
from glob import glob
import random
from tensorboardX import SummaryWriter
import torch 
import time
import datetime
from eval import val
from utils import *
from training.losses import *
from training.show_gpu import showGPU
from training.train_ing_func import make_models_dict, update_best_model
from einops import rearrange


def training(cfg, dataset, dataloader, models, losses, opts, schs, scores):
    # define start_iter
    start_iter = scores['step'] if scores['step'] > 0 else 0

    # find epoch 
    epoch = int(scores['step']/len(dataloader)) # [epoch: current step / (data size / batch size)] ex) 8/(16/4)
    start_epoch = epoch

    # define writer
    writer = SummaryWriter(f'tensorboard_log/{cfg.dataset}_bs{cfg.batch_size}')

    # flownet, segnet
    flownet = models['flownet']
    segnet = models['segnet']

    # training start!
    training = True  
    torch.autograd.set_detect_anomaly(True)

    print('\n===========================================================')
    print('Training Start!')
    print('===========================================================')

    while training:
            # GPU Status Check!
            if epoch == start_epoch+1:
                showGPU()
            
            '''
            ------------------------------------
            Training (1 epoch)
            ------------------------------------
            '''
            for indice, clips in dataloader:
                # define frame 1 to 4 
                frame_1 = clips[:, 0:3, :, :].cuda()  # (n, 3, 256, 256) 
                frame_2 = clips[:, 3:6, :, :].cuda()  # (n, 3, 256, 256) 
                frame_3 = clips[:, 6:9, :, :].cuda()  # (n, 3, 256, 256) 
                frame_4 = clips[:, 9:12, :, :].cuda()  # (n, 3, 256, 256) 

                # define flow 1 to 4
                flow_1_input = torch.cat([frame_1.unsqueeze(2), frame_2.unsqueeze(2)], 2).cuda() # (n, 3, 2, 256, 256)
                flow_1 = (flownet(flow_1_input * 255.) / 255.).detach().cuda() # (n, 2, 256, 256)
                flow_2_input = torch.cat([frame_2.unsqueeze(2), frame_3.unsqueeze(2)], 2).cuda() # (n, 3, 2, 256, 256)
                flow_2 = (flownet(flow_2_input * 255.) / 255.).detach().cuda() # (n, 2, 256, 256)
                flow_3_input = torch.cat([frame_3.unsqueeze(2), frame_4.unsqueeze(2)], 2).cuda() # (n, 3, 2, 256, 256)
                flow_3 = (flownet(flow_3_input * 255.) / 255.).detach().cuda() # (n, 2, 256, 256)

                # define label
                batch_size = frame_1.shape[0]
                for i in range(batch_size):
                    with torch.no_grad():
                        seg_input = torch.cat([frame_1[i].unsqueeze(0), frame_2[i].unsqueeze(0), frame_3[i].unsqueeze(0), frame_4[i].unsqueeze(0)], 0).cuda() # (4, 3, 256, 256)
                        seg_output = segnet(seg_input)['out'].cuda() # (4, 21, 256, 256)
                        seg_output = rearrange(seg_output, 'b c h w -> (b c) h w').unsqueeze(0).cuda() # (1, 84, 256, 256)
                    if i == 0:
                        label_input = seg_output.cuda() # (1, 84, 256, 256)
                    else:
                        label_input = torch.cat([label_input, seg_output], 0).cuda() # (n, 84, 256, 256)

                # pop() the used video index
                for index in indice:
                    dataset.all_seqs[index].pop()
                    if len(dataset.all_seqs[index]) == 0:
                        dataset.all_seqs[index] = list(range(len(dataset.videos[index]) - 4))
                        random.shuffle(dataset.all_seqs[index])

                '''
                -------------------
                Forward 
                -------------------
                '''
                # Forward Input / Target
                f_input1 = torch.cat([frame_1,frame_2, frame_3, frame_4], 1).cuda() # (n, 12, 256, 256) 
                f_input2 = torch.cat([flow_1, flow_2, flow_3], 1).cuda() # (n, 6, 256, 256)
                f_input3 = label_input.cuda() # (n, 84, 256, 256)
                f_target = clips[:, 12:15, :, :].cuda()  # (n, 3, 256, 256) 

                # Forward
                f_gl, f_dl, FG_frame = first_direction(cfg=cfg, input1=f_input1, input2=f_input2, input3=f_input3, target=f_target, models=models, losses=losses) # (n, 3, 256, 256) 

                G_l = f_gl
                D_l = f_dl

                # weight update 
                opts['optimizer_G'].zero_grad()
                G_l.backward()
                opts['optimizer_G'].step()
                
                opts['optimizer_D'].zero_grad()
                D_l.backward()
                opts['optimizer_D'].step()

                # calculate time
                torch.cuda.synchronize()
                time_end = time.time()
                if scores['step'] > start_iter:  
                    iter_t = time_end - temp
                temp = time_end

                if scores['step'] != start_iter:
                    '''
                    ------------------------------------
                    check train status per 20 iteration!
                    ------------------------------------
                    '''
                    if scores['step'] % 20 == 0:
                        print(f"===========epoch:{epoch} (step:{scores['step']})============")

                        # calculate remained time
                        time_remain = (cfg.iters - scores['step']) * iter_t
                        eta = str(datetime.timedelta(seconds=time_remain)).split('.')[0]
                        
                        # calculate psnr
                        f_psnr = psnr_error(FG_frame, f_target)

                        # current lr
                        lr_g = opts['optimizer_G'].param_groups[0]['lr']
                        lr_d = opts['optimizer_D'].param_groups[0]['lr']

                        # show psnr, loss, time, lr
                        print(f"[{scores['step']}] G_l: {G_l:.3f} | D_l: {D_l:.3f} | psnr(g): {f_psnr:.3f} | "\
                            f"best_auc(g): {scores['g_best_auc']:.3f} | "\
                            f"iter_t: {iter_t:.3f}s | remain_t: {eta} | lr_g: {lr_g:.7f} | lr_d: {lr_d:.7f}")
                        
                        # write psnr, loss, lr
                        writer.add_scalar('psnr/train_psnr', f_psnr, global_step=scores['step'])
                        writer.add_scalar('G_loss_total/g_loss', G_l, global_step=scores['step'])
                        writer.add_scalar('D_loss_total/d_loss', D_l, global_step=scores['step'])
                        writer.add_scalar('lr/lr_g', lr_g, global_step=scores['step'])
                        writer.add_scalar('lr/lr_d', lr_d, global_step=scores['step'])

                    '''
                    -------------------------------
                    early stopping per val_interval 
                    --------------------------------
                    '''
                    if scores['step'] % cfg.val_interval == 0:
                        _, g_auc, scores = val(cfg=cfg, train_scores=scores, models=models, iter=scores['step'])

                        # find Best AUC
                        update_best_model('g_best_auc', g_auc, 'g', scores['step'], cfg, models, opts, scores)

                    '''
                    ------------------------------------
                    save current model per save_interval 
                    ------------------------------------
                    '''
                    if scores['step'] % cfg.save_interval == 0:
                        model_dict = make_models_dict(models, opts, scores)
                        torch.save(model_dict, f'weights/{cfg.work_num}_latest_{cfg.dataset}.pth')
                        print(f"\nAlready saved: \'{cfg.work_num}_latest_{cfg.dataset}.pth\'.")
                    
                    '''
                    ------------------------------------
                    save last model
                    ------------------------------------
                    '''
                    if scores['step'] == cfg.iters:
                        training = False
                        model_dict = make_models_dict(models, opts, scores)
                        torch.save(model_dict, f"weights/{cfg.work_num}_last_{cfg.dataset}.pth")
                        break 

                # one iteration ok!
                scores['step'] += 1
            
            # one epoch ok!
            epoch += 1
            if cfg.sch:
                schs['sch_G'].step()
                schs['sch_D'].step()


def first_direction(cfg, input1, input2, input3, target, models, losses):
    '''
    Return generator_loss, discriminator_loss, generated_frame
    '''
    generator = models['generator']
    discriminator = models['discriminator']

    discriminate_loss = losses['discriminate_loss']
    intensity_loss = losses['intensity_loss']
    gradient_loss = losses['gradient_loss']
    adversarial_loss = losses['adversarial_loss']
    triplet_loss = losses['triplet_loss']

    coefs = [1, 1, 0.05, 1] # inte_l, grad_l, adv_l, tri_l

    # generator prediction and get loss
    pred, frame_feature, motion_feature, ftom_feature, label_feature, ftol_feature = generator(input1, input2, input3)
    inte_l = intensity_loss(pred, target)
    grad_l = gradient_loss(pred, target)
    adv_l = adversarial_loss(discriminator(pred))
    tri_l = triplet_loss(anchor=ftom_feature, positive=motion_feature, negative=frame_feature) + \
            triplet_loss(anchor=ftol_feature, positive=label_feature, negative=frame_feature) 

    loss_gen = coefs[0] * inte_l + \
                coefs[1] * grad_l + \
                coefs[2] * adv_l + \
                coefs[3] * tri_l

    # discriminator
    loss_dis = discriminate_loss(discriminator(target),
                                 discriminator(pred.detach()))

    return loss_gen, loss_dis, pred

