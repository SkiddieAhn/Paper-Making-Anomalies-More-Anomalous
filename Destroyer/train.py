import argparse
from training.train_pre_func import *
from training.train_func import *
import Dataset
from torch.utils.data import DataLoader
from config import update_config

def main():
    parser = argparse.ArgumentParser(description='MAMA_Destroyer')
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--dataset', default='avenue', type=str, help='The name of the dataset to train.')
    parser.add_argument('--iters', default=60000, type=int, help='The total iteration number.')
    parser.add_argument('--resume', default=None, type=str,
                        help='The pre-trained model to resume training with, pass \'latest\' or the model name.')
    parser.add_argument('--save_interval', default=1000, type=int, help='Save the model every [save_interval] iterations.')
    parser.add_argument('--val_interval', default=1000, type=int, help='Evaluate the model every [val_interval] iterations')
    parser.add_argument('--work_num', default=0, type=int)
    parser.add_argument('--save_dir', default='sha', type=str, help='model save directory')
    parser.add_argument('--sch', default=False, type=str2bool, nargs='?', const=True, help='scheduler')
    parser.add_argument('--manualseed', default=-1, type=int, help='manual seed')

    args = parser.parse_args()
    train_cfg = update_config(args, mode='train')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    print(device)

    '''
    <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    Pre-work for Training
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    '''
    
    # setup seed (for deterministic behavior)
    seed(seed_value=train_cfg.manualseed)

    # get dataset and loader
    train_dataset = Dataset.train_dataset(train_cfg)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_cfg.batch_size, shuffle=True, num_workers=4, drop_last=True)
    print_infor(cfg=train_cfg, dataloader=train_dataloader)

    # define models
    generator, autoencoder, flownet, segnet = def_models()

    # define optimizer and scheduler
    optimizer_A, sch_A = def_optim_sch(train_cfg, autoencoder)

    # load models
    load_models(train_cfg, generator, autoencoder, flownet, segnet)

    # load scores
    scores = load_scores()

    # make dict
    models = make_model_dict(generator, autoencoder, flownet, segnet)
    opts = make_opt_dict(optimizer_A)
    schs = make_sch_dict(train_cfg, sch_A)


    '''
    <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    Training
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    '''

    # train
    training(train_cfg, train_dataset, train_dataloader, models, opts, schs, scores)


if __name__=="__main__":
    main()