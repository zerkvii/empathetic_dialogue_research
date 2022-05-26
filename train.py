from typing import Dict, Optional
import argparse
from ast import arg
from dis import dis
import random
from turtle import forward
import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from generator import BeamSearchGenerator, DBSGenerator, GreedyGenerator
from lightning_model import AFModel
from optimizer import OpenAIAdam
from configs import DEFAULT_GEN_CONFIG, DEFAULT_MODEL_CFG, DEFAULT_OPT_CFG
from model import AdmELM, ELMModel, LMModel, MtlELM, load_openai_pretrained_model
from data_loader import load_dataset
from utils import cal_clf_acc, dotdict, make_infinite, moses_multi_bleu, stack_input, make_path, \
    get_time_str, Logger, delete_file, count_parameters, get_available_gpu
from data_loader import load_dataset
from utils import make_infinite, stack_input, make_path, \
    get_time_str, Logger, delete_file, count_parameters
from time import time
from indexer import Indexer
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
from pytorch_lightning import LightningModule, seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
available_devices, devices = get_available_gpu()
os.environ['CUDA_VISIBLE_DEVICES'] = devices
os.environ["TOKENIZERS_PARALLELISM"] = "false"




def parse_args():
    parser = argparse.ArgumentParser()
    # model configs
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--n_emo_embd', type=int, default=768)
    parser.add_argument('--init_std', type=float, default=0.02)
    parser.add_argument('--tieSL', default=False, action='store_true')
    # training configs
    parser.add_argument('--n_epoch', type=int, default=3)
    parser.add_argument('--n_batch', type=int, default=8)
    parser.add_argument('--max_patience', type=int, default=3)
    # other configs
    parser.add_argument('--dev', type=bool, default=False)
    parser.add_argument('--test', type=bool, default=False)
    parser.add_argument('--model_name', type=str, default='trans')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_pretrained', default=False, action='store_true')
    parser.add_argument('--pretrained_dir', type=str,
                        default='save/pretrained_lm/')
    parser.add_argument('--testid_filter_path', type=str, default='empdial_dataset/testset_idxs_5248.npy')
    parser.add_argument('--testid_sample_path', type=str)
    # generation configs
    parser.add_argument('--max_gen_len', type=int, default=50)
    parser.add_argument('--beam_size', type=int, default=5)
    parser.add_argument('--dbs_beam_size', type=int, default=1)
    parser.add_argument('--dbs_groups', type=int, default=5)
    parser.add_argument('--dbs_lambda', type=int, default=0.5)
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # batch size
    batch_size = args.n_batch
    # model configs
    model_cfg = DEFAULT_MODEL_CFG
    opt_cfg = DEFAULT_OPT_CFG
    #generator config
    gen_cfg=DEFAULT_GEN_CONFIG
    # indexer
    indexer = Indexer(model_cfg.n_ctx)

    trainset, data_loader_train = load_dataset(
        'train', 
        indexer,
        batch_size)
    devset, data_loader_dev = load_dataset(
        'dev',
        indexer,
        batch_size,
        shuffle=False)
    testset, data_loader_test = load_dataset(
        'dev',
        indexer,
        batch_size=1,
        shuffle=False)
    # to avoid memory overflow
    trainset.filter_max_len(indexer.n_ctx, 'train')
    devset.filter_max_len(indexer.n_ctx, 'dev')

    testset.filter_by_idxs(np.load(args.testid_filter_path))
    if args.testid_sample_path is not None:
        testset.filter_by_idxs(np.load(args.testid_sample_path))
    
    
    tr_iter = make_infinite(data_loader_train)
    n_epoch = args.n_epoch
    n_iter = int(np.ceil(len(trainset) / batch_size)) * n_epoch
    # create and load pretrained model
    model = AFModel(model_name=args.model_name,
                    indexer=indexer,
                    model_cfg=dict(model_cfg),
                    opt_cfg=dict(opt_cfg),
                    gen_cfg=dict(gen_cfg),
                    beta=args.beta,
                    init_std=args.init_std,
                    tieSL=args.tieSL,
                    n_iter=n_iter,
                    )
    if not args.no_pretrained:
        load_openai_pretrained_model(model.model.transformer, model_cfg,
                                     n_special=indexer.n_special,
                                     dir=args.pretrained_dir)

    #################### training ####################

    checkpoint_callback = ModelCheckpoint(
        monitor="val_ppl", filename=f"{args.model_name}", mode="min")
    trainer = Trainer(max_epochs=n_epoch,
                      accelerator="gpu",
                      devices=available_devices,
                      strategy='ddp',
                      callbacks=[checkpoint_callback]
                      )
    if args.dev:
        trainer = Trainer(max_epochs=5,
                          accelerator="gpu",
                          devices=1,
                          )
    if not args.test:
        trainer.fit(model=model, train_dataloaders=data_loader_train,
                    val_dataloaders=data_loader_dev)

    if args.test:
        version_num = 0
        checkpoints_file = f'./lightning_logs/version_{version_num}/checkpoints/adde_model.ckpt'
        model.load_from_checkpoint(checkpoints_file)
        trainer.test(model=model,dataloaders=data_loader_test)
