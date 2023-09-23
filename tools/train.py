import os
from os.path import join
import sys
import argparse

sys.path.append("/ReZSL-github")

import torch
import numpy as np
import random

from REZSL.data import build_dataloader
from REZSL.modeling import build_zsl_pipeline, ReZSL, get_attributes_info
from REZSL.solver import make_optimizer, make_lr_scheduler
from REZSL.engine.trainer import do_train

from REZSL.config import cfg
from REZSL.utils.comm import *

from REZSL.utils import ReDirectSTD, set_seed

#from apex import amp

def train_model(cfg, local_rank, distributed):
    device = cfg.MODEL.DEVICE
    model = build_zsl_pipeline(cfg)
    tr_dataloader, tu_loader, ts_loader, res = build_dataloader(cfg, is_distributed=distributed)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer, len(tr_dataloader))

    use_mixed_precision = cfg.DTYPE == "float16"
    #amp_opt_level = 'O1' if use_mixed_precision else 'O0'
    #model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)

    model = torch.nn.DataParallel(model, device_ids=cfg.MODEL.GPUS).to(device)

    output_dir = cfg.OUTPUT_DIR
    model_file_name = cfg.MODEL_FILE_NAME
    model_file_path = join(output_dir, model_file_name)

    test_gamma = cfg.TEST.GAMMA
    max_epoch = cfg.SOLVER.MAX_EPOCH
    use_REZSL = cfg.MODEL.REZSL.USE
    WeightType = cfg.MODEL.REZSL.WEIGHT_TYPE

    RegNorm = cfg.MODEL.LOSS.REG_NORM
    RegType = cfg.MODEL.LOSS.REG_TYPE
    scale = cfg.MODEL.SCALE

    info = get_attributes_info(cfg.DATASETS.NAME, cfg.DATASETS.SEMANTIC_TYPE)
    attritube_num = info["input_dim"]
    cls_num = info["n"]
    ucls_num = info["m"]
    scls_num = cls_num - ucls_num

    rezsl = ReZSL(p=cfg.MODEL.REZSL.P, p2=cfg.MODEL.REZSL.P2, att_dim=attritube_num, train_class_num=scls_num,
                  test_class_num=cls_num, RegNorm=RegNorm, RegType=RegType, WeightType=WeightType, device=device)

    lamd = {
        0: cfg.MODEL.LOSS.LAMBDA0,
        1: cfg.MODEL.LOSS.LAMBDA1,
        2: cfg.MODEL.LOSS.LAMBDA2,
        3: cfg.MODEL.LOSS.LAMBDA3,
    }

    do_train(
        model,
        rezsl,
        tr_dataloader,
        tu_loader,
        ts_loader,
        res,
        optimizer,
        scheduler,
        lamd,
        test_gamma,
        use_REZSL,
        RegNorm,
        RegType,
        scale,
        device,
        max_epoch,
        model_file_path,
        cfg
    )

    return model


def main():
    parser = argparse.ArgumentParser(description="PyTorch Zero-Shot Learning Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    args.distributed = num_gpus > 1
    print("distributed?")
    print(args.distributed)

    seed = cfg.SEED
    set_seed(seed)

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.freeze()


    output_dir = cfg.OUTPUT_DIR
    log_file_name = cfg.LOG_FILE_NAME

    log_file_path = join(output_dir, log_file_name)

    if is_main_process():
        ReDirectSTD(log_file_path, 'stdout', True)

    print("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        print(config_str)
    print("Running with config:\n{}".format(cfg))

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    model = train_model(cfg, args.local_rank, args.distributed)

if __name__ == '__main__':
    main()
