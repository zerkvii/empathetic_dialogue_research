import argparse
from ast import arg
from dis import dis
import random
from turtle import forward
import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from optimizer import OpenAIAdam
from configs import DEFAULT_MODEL_CFG, DEFAULT_OPT_CFG
from model import LMModel, load_openai_pretrained_model
from data_loader import load_dataset, load_dataset_ddp
from utils import dotdict, make_infinite, stack_input, make_path, \
    get_time_str, Logger, delete_file, count_parameters, get_available_gpu
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
from typing import Dict
os.environ['CUDA_VISIBLE_DEVICES'] = devices
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class AFModel(LightningModule):
    def __init__(self, model_name,
                 indexer:Indexer,
                 model_cfg:Dict,
                 opt_cfg:Dict,
                 n_iter: int = 888,
                 ) -> None:
        super().__init__()
        self.indexer=indexer
        self.save_hyperparameters(ignore=['indexer'])

        # self.save_hyperparameters(ignore=['indexer'])
        self.model = LMModel(dotdict(model_cfg), self.indexer.n_vocab, self.indexer.n_special, self.indexer.n_ctx)

    def compute_batch_loss(self, batch):
        # stack token, dialog states and position encoding
        X = stack_input(batch['dialog'], [batch['dialog_state']], self.indexer)
        # X = X.to(device)
        # compute LM logits and loss
        lm_logits, _ = self(X)
        # mask = batch['dialog_mask'].to(device)
        mask = batch['dialog_mask']
        # calculate language modelling loss
        target_shifted = X[:, 1:, 0].contiguous().view(-1)
        lm_logits_shifted = lm_logits[:, :-1, :]
        lm_logits_shifted = lm_logits_shifted.contiguous().view(-1,
                                                                lm_logits.shape[-1])
        loss = F.cross_entropy(
            lm_logits_shifted, target_shifted, reduction='none')
        mask_shifted = mask[:, 1:]
        loss = torch.sum(loss.view(mask_shifted.shape) *
                         mask_shifted) / torch.sum(mask_shifted)
        return loss

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss = self.compute_batch_loss(batch)
        self.log('train_loss',loss)
        self.log('train_ppl',torch.exp(loss))
        return loss

    def validation_step(self, batch, batch_idx):
        val_loss = self.compute_batch_loss(batch)

        self.log('val_loss',val_loss)
        self.log('val_ppl',torch.exp(val_loss))
        return val_loss

    def configure_optimizers(self):
        cfg=dotdict(self.hparams.opt_cfg)
        model_opt = OpenAIAdam(self.parameters(),
                               t_total=self.hparams.n_iter,
                               lr=cfg.lr,
                               schedule=cfg.lr_schedule,
                               warmup=cfg.lr_warmup,
                               b1=cfg.b1,
                               b2=cfg.b2,
                               e=cfg.e,
                               l2=cfg.l2,
                               vector_l2=cfg.vector_l2,
                               max_grad_norm=cfg.max_grad_norm)

        return model_opt


def parse_args():
    parser = argparse.ArgumentParser()
    # training configs
    parser.add_argument('--n_epoch', type=int, default=15)
    parser.add_argument('--n_batch', type=int, default=32)
    parser.add_argument('--max_patience', type=int, default=3)
    # other configs
    parser.add_argument('--dev', type=bool, default=False)
    parser.add_argument('--test', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_pretrained', default=False, action='store_true')
    parser.add_argument('--pretrained_dir', type=str,
                        default='save/pretrained_lm/')
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
    opt_cfg=DEFAULT_OPT_CFG
    # indexer
    indexer = Indexer(model_cfg.n_ctx)

    trainset, data_loader_train = load_dataset(
        'train', indexer, batch_size)
    devset, data_loader_dev = load_dataset('dev', indexer, batch_size)
    # to avoid memory overflow
    trainset.filter_max_len(indexer.n_ctx, 'train')
    devset.filter_max_len(indexer.n_ctx, 'dev')


    tr_iter = make_infinite(data_loader_train)
    n_epoch = args.n_epoch
    n_iter = int(np.ceil(len(trainset) / batch_size)) * n_epoch

    # create and load pretrained model
    model = AFModel(model_name='adde-lm',
                    indexer=indexer,
                    model_cfg=dict(model_cfg),
                    opt_cfg=dict(opt_cfg),
                    n_iter=n_iter,
                    )
    if not args.no_pretrained:
        load_openai_pretrained_model(model.model.transformer, model_cfg,
                                     n_special=indexer.n_special,
                                     dir=args.pretrained_dir)


    #################### training ####################

    trainer=Trainer
    checkpoint_callback = ModelCheckpoint(
        monitor="val_ppl", filename=f"adde_model", mode="min")
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
        trainer.fit(model=model, train_dataloaders=data_loader_train, val_dataloaders=data_loader_dev)
    
    if args.test:
        version_num=0
        checkpoints_file=f'./lightning_logs/version_{version_num}/checkpoints/adde_model.ckpt'
        model.load_from_checkpoint(checkpoints_file)
