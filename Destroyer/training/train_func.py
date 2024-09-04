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
from training.train_ing_func import np_load_frame, make_models_dict, update_best_model
from training.noise import * 
from torchvision.utils import save_image


def training(cfg, dataset, dataloader, models, opts, schs, scores):
    # define start_iter
    start_iter = scores['step'] if scores['step'] > 0 else 0

    # find epoch 
    epoch = int(scores['step']/len(dataloader)) # [epoch: current step / (data size / batch size)] ex) 8/(16/4)
    start_epoch = epoch

    # flownet, segnet
    flownet = models['flownet']
    segnet = models['segnet']

    # define patchGenerator
    pg = PatchGenerator(img_size=256, patch_size=32)

    # define bg tensor
    b_path = f'training/background/black.jpg'
    bg = np_load_frame(b_path, 256, 256).cuda() # [c, h, w]
    bg = bg.unsqueeze(0).repeat(cfg.batch_size, 1, 1, 1) # [b, c, h, w]

    # define writer
    writer = SummaryWriter(f'tensorboard_log/{cfg.dataset}_bs{cfg.batch_size}')

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
                _, _, FG_frame = first_direction(input1=f_input1, input2=f_input2, input3=f_input3, models=models) # (n, 3, 256, 256) 

                '''
                -------------------
                Destroy Process
                -------------------
                '''
                # Destroy with AE
                g_output = FG_frame.detach()
                a_input, _ = randomNoisePatches(pg=pg, input=g_output.clone(), method=2)
                A_l, A_frame = second_direction(pg=pg, A=g_output, N=a_input, GT=f_target, BG=bg, models=models)

                # weight update 
                opts['optimizer_A'].zero_grad()
                A_l.backward()
                opts['optimizer_A'].step()

                # calculate time
                torch.cuda.synchronize()
                time_end = time.time()
                if scores['step'] > start_iter:  
                    iter_t = time_end - temp
                temp = time_end

                '''
                ------------------------------------
                Visualization with Training!
                ------------------------------------
                '''
                if (epoch+1) % 1000 == 0: 
                    if not os.path.exists(f"train_img/{epoch}"):
                        os.makedirs(f"train_img/{epoch}")

                    for i in range(4):
                        g_output_ = ((FG_frame[i] + 1 ) / 2)[(2,1,0),...]
                        g_target_ = ((f_target[i] + 1 ) / 2)[(2,1,0),...]
                        a_input_ = ((a_input[i] + 1 ) / 2)[(2,1,0),...]
                        a_output_ = ((A_frame[i] + 1 ) / 2)[(2,1,0),...]

                        save_image(g_output_, f'train_img/{epoch}/generator_output_{i}.jpg')
                        save_image(g_target_, f'train_img/{epoch}/generator_target_{i}.jpg')
                        save_image(a_input_, f'train_img/{epoch}/autoencoder_input_{i}.jpg')
                        save_image(a_output_, f'train_img/{epoch}/autoencoder_output_{i}.jpg')
                        
                        
                        print('==================================================================================')
                        print(f'[{epoch}] save image ok!')
                        print('==================================================================================')
                    
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
                        a_psnr = psnr_error(a_input, A_frame)

                        # current lr
                        lr_a = opts['optimizer_A'].param_groups[0]['lr']

                        # show psnr, loss, time, lr
                        print(f"[{scores['step']}] A_l: {A_l:.3f} | psnr(g): {f_psnr:.3f} | psnr(a): {a_psnr:.3f} | "\
                              f"best_auc(g): {scores['g_best_auc']:.3f} | best_auc(a): {scores['a_best_auc']:.3f} | iter_t: {iter_t:.3f}s | remain_t: {eta}")
                                                
                        # write psnr, loss, lr
                        writer.add_scalar('psnr/train_psnr', f_psnr, global_step=scores['step'])
                        writer.add_scalar('A_loss_total/a_loss', A_l, global_step=scores['step'])
                        writer.add_scalar('lr/lr_a', lr_a, global_step=scores['step'])

                    '''
                    -------------------------------
                    early stopping per val_interval 
                    --------------------------------
                    '''
                    if scores['step'] % cfg.val_interval == 0:
                        _, g_auc, a_auc, scores = val(cfg=cfg, train_scores=scores, models=models, iter=scores['step'])

                        # find Best AUC
                        update_best_model('g_best_auc', g_auc, 'g', scores['step'], cfg, models, opts, scores)
                        update_best_model('a_best_auc', a_auc, 'a', scores['step'], cfg, models, opts, scores)

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
                schs['sch_A'].step()


def first_direction(input1, input2, input3, models):
    '''
    Return generator_loss, discriminator_loss, generated_frame
    '''
    generator = models['generator']

    # generator prediction and get loss
    pred, _, _, _, _, _ = generator(input1, input2, input3)

    return 0, 0, pred


def second_direction(pg, A, N, GT, BG, models):
    autoencoder = models['autoencoder']

    destroyer_loss = getDestroyerLoss(pg, ld=4)

    # autoencoder prediction and get loss
    G = autoencoder(N)
    loss = destroyer_loss(A=A, N=N, GT=GT, G=G, BG=BG)

    return loss, G