import argparse
import os.path as osp
import shutil
import time
import torch
from vedacore.misc import Config, mkdir_or_exist, set_random_seed
from vedacore.parallel import init_dist
from vedatad.assembler import trainval
from vedatad.misc import get_root_logger
import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--workdir', help='the dir to save logs and models')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)  # TODO

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)

    cfg = Config.fromfile(args.config)

    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # workdir is determined in this priority: CLI > segment in file > filename
    if args.workdir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.workdir = args.workdir
    elif cfg.get('workdir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.workdir = osp.join('./workdir',
                               osp.splitext(osp.basename(args.config))[0])

    seed = cfg.get('seed', None)
    deterministic = cfg.get('deterministic', False)
    set_random_seed(seed, deterministic)

    # create work_dir
    mkdir_or_exist(osp.abspath(cfg.workdir))
    shutil.copy(args.config, cfg.workdir)
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.workdir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
    max_gpu_mem = cfg.get("max_gpu_mem", -1)
    if max_gpu_mem != -1:
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024/1024/1024
        fraction = max_gpu_mem / total_mem
        torch.cuda.set_per_process_memory_fraction(fraction)
    torch.autograd.set_detect_anomaly(True)

    trainval(cfg, distributed, logger)


if __name__ == '__main__':
    main()
