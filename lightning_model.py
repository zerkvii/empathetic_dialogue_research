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

class AFModel(LightningModule):
    def __init__(self, model_name,
                 indexer: Indexer,
                 model_cfg: Dict,
                 opt_cfg: Dict,
                 gen_cfg:Dict,
                 beta: float,
                 init_std: float,
                 tieSL: bool,
                 n_iter: int = 888,
                 ) -> None:
        super().__init__()
        self.indexer = indexer
        self.save_hyperparameters(ignore=['indexer'])

        # self.save_hyperparameters(ignore=['indexer'])
        if self.hparams.model_name == 'trans':
            self.model = LMModel(dotdict(
                model_cfg), self.indexer.n_vocab, self.indexer.n_special, self.indexer.n_ctx)
        elif self.hparams.model_name == 'adde':
            self.model = ELMModel(dotdict(model_cfg),
                                  self.indexer.n_vocab,
                                  self.indexer.n_special,
                                  self.indexer.n_ctx,
                                  self.indexer,
                                  beta=beta,
                                  init_std=init_std,
                                  tieSL=tieSL
                                  )
        elif self.hparams.model_name == 'adm':
            self.model = AdmELM(dotdict(model_cfg),
                                self.indexer.n_vocab,
                                self.indexer.n_special,
                                self.indexer.n_ctx,
                                self.indexer,
                                beta=beta,
                                init_std=init_std,
                                tieSL=tieSL
                                )
        elif self.hparams.model_name == 'mtl':
            self.model = MtlELM(dotdict(model_cfg), indexer.n_vocab,
                                indexer.n_special, indexer.n_ctx, indexer)

    def compute_batch_loss(self, batch):
        # stack token, dialog states and position encoding
        X = stack_input(batch['dialog'], [batch['dialog_state']], self.indexer)
        # X = X.to(device)

        # compute LM logits and loss
        if self.hparams.model_name == 'trans':
            logits, _ = self.model(X)
        elif self.hparams.model_name == 'adde':
            logits, _ = self.model(batch['emotion'], X)
        elif self.hparams.model_name in ['adm', 'mtl']:
            logits, _, clf_logits = self.model(batch['clf_idx'], X)
        # mask = batch['dialog_mask'].to(device)
        mask = batch['dialog_mask']
        # calculate language modelling loss
        target_shifted = X[:, 1:, 0].contiguous().view(-1)
        logits_shifted = logits[:, :-1, :]
        logits_shifted = logits_shifted.contiguous().view(-1, logits.shape[-1])
        loss = F.cross_entropy(
            logits_shifted, target_shifted, reduction='none')
        mask_shifted = mask[:, 1:]
        loss = torch.sum(loss.view(mask_shifted.shape) *
                         mask_shifted) / torch.sum(mask_shifted)
        if self.hparams.model_name in ['adm', 'mtl']:
            emo_label = batch['emotion']
            clf_loss = F.cross_entropy(clf_logits, emo_label, reduction='mean')
            # calculate emotion clf accuracy
            acc_top1, acc_top5 = cal_clf_acc(clf_logits, emo_label.tolist())
            # joint_loss = loss  clf_loss
            # return joint_loss
            return loss, clf_loss, (acc_top1, acc_top5)
        return loss

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # test batch
        # if batch_idx%20==0:
        # print(self.indexer.decode_text_sequence(batch['dialog'][0]))
        # print(self.indexer.decode_text_sequence(batch['dialog_state'][0]))
        if self.hparams.model_name in ['adm', 'mtl']:
            loss, clf_loss, _ = self.compute_batch_loss(batch)
            joint_loss = loss+clf_loss
            self.log('train_loss', joint_loss)
            self.log('train_clf_loss', clf_loss)
            self.log('train_ppl', torch.exp(loss))
            return joint_loss
        else:
            loss = self.compute_batch_loss(batch)
            self.log('train_loss', loss)
            self.log('train_ppl', torch.exp(loss))
            return loss

    def validation_step(self, batch, batch_idx):
        if self.hparams.model_name in ['adm', 'mtl']:
            val_lm_loss, val_clf_loss, (val_acc_1, val_acc_5) = self.compute_batch_loss(
                batch)
            val_joint_loss = val_lm_loss + val_clf_loss
            self.log('val_loss', val_joint_loss)
            self.log('val_ppl', torch.exp(val_lm_loss))
            self.log('val_acc_1', val_acc_1)
            self.log('val_acc_5', val_acc_5)
            self.log('clf_loss', val_clf_loss)
            return val_joint_loss
        else:
            loss = self.compute_batch_loss(batch)
            self.log('val_loss', loss)
            self.log('val_ppl', torch.exp(loss))
            return loss

    def test_step(self, batch, batch_idx):
        logstr = []
        logstr.append('[Context]:')
        for c in batch['data'][0]['context_text']:
                logstr.append(' - ' + c)
        logstr.append('[Golden]:')
        golden = batch['data'][0]['target_text']
        logstr.append(' - ' + golden)
        self.resp_golden.append(golden)
        # beam search
        logstr.append('[Beam=%d]:' % self.hparams.gen_cfg.beam_size)
        beam_resp, _ = self.generator.generate(batch['dialog'], batch['dialog_state'])
        self.resp_beam.append(beam_resp)
        logstr.append(' - ' + beam_resp)
        logstr = str.join('\n', logstr)
        tensorboard = self.logger.experiment
        tensorboard.add_text('logstr',logstr)

    def test_epoch_end(self, outputs) -> None:
        resp_beam = np.array(self.resp_beam)
        # resp_dbs = np.array(self.resp_dbs)
        resp_golden = np.array(self.resp_golden)
        # BLEU
        # bleu_greedy = moses_multi_bleu(resp_greedy, resp_golden, lowercase=True)
        bleu_beam = moses_multi_bleu(resp_beam, resp_golden, lowercase=True)
        # bleu_dbs = moses_multi_bleu(resp_dbs, resp_golden, lowercase=True)
        self.log('\n' + '-'*10 + '\n')
        self.log('BLEU beam: %.3f' % bleu_beam)
        # logger.log('BLEU greedy: %.3f, BLEU beam: %.3f, BLEU DBS: %.3f' % \
        #            (bleu_greedy, bleu_beam, bleu_dbs))
        # save outputs
        # np.save(os.path.join(args.log_dir, 'greedy.npy'), resp_greedy)
        np.save(os.path.join(self.hparams.gen_cfg.log_dir, 'beam.npy'), resp_beam)
    
    def setup(self, stage: Optional[str] = 'test') -> None:
        if stage == 'test':
            # self.generator = GreedyGenerator(self.model, self.hparams.gen_cfg.max_gen_len, self.indexer)
            self.generator = BeamSearchGenerator(self.model, self.hparams.gen_cfg.max_gen_len,
                                                 self.indexer, self.hparams.gen_cfg.beam_size)
            # self.generator = DBSGenerator(self.model, self.hparams.gen_cfg.max_gen_len,
            #                               self.indexer, self.hparams.gen_cfg.dbs_beam_size,
            #                               self.hparams.gen_cfg.dbs_groups,
            #                               self.hparams.gen_cfg.dbs_lambda)
            self.resp_beam = []
            # self.resp_dbs = []
            # self.dbs_gids = []
            self.resp_golden = []
    def configure_optimizers(self):
        cfg = dotdict(self.hparams.opt_cfg)
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