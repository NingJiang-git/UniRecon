import logging
import torch
from os import path as osp
import argparse
import random
from data import create_dataloader, create_dataset
from models import create_model
from utils import (get_root_logger, get_time_str,
                           make_exp_dirs)
from utils import (MessageLogger, check_resume,
                           get_root_logger, get_time_str, init_tb_logger,
                           init_wandb_logger, make_exp_dirs, mkdir_and_rename,
                           set_random_seed)
from utils.dist_util import get_dist_info, init_dist
from utils.options import dict2str, parse

def parse_options(is_train=False):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = parse(args.opt, is_train=is_train)

    # distributed settings
    if args.launcher == 'none':
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)
            print('init dist .. ', args.launcher)

    opt['rank'], opt['world_size'] = get_dist_info()

    # random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    return opt

def main():
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=False)

    torch.backends.cudnn.benchmark = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'],
                        f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    # logger.info(get_env_info())
    logger.info(dict2str(opt))

    # create test dataset and dataloader
    test_loaders = []
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(
            test_set,
            dataset_opt,
            num_gpu=opt['num_gpu'],
            dist=opt['dist'],
            sampler=None,
            seed=opt['manual_seed'])
        logger.info(
            f"Number of test images in {dataset_opt['data_file']}: {len(test_set)}")
        test_loaders.append(test_loader)

    # create model
    model = create_model(opt)


    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['data_file']
        logger.info(f'Testing {test_set_name}...')
        model.validation(
            test_loader,
            current_iter=opt['name'],
            tb_logger=None,)


if __name__ == '__main__':
    main()
