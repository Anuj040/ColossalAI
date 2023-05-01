from abc import ABC
from datetime import datetime
from typing import Optional

import pandas as pd
import torch
import torch.distributed as dist
from torch.optim import Optimizer, lr_scheduler
from transformers.trainer import get_scheduler
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
import math
from .strategies import Strategy
from .utils import is_rank_0


class RewardModelTrainer(ABC):
    """
        Trainer to use while training reward model.

    Args:
        model (torch.nn.Module): the model to train
        strategy (Strategy): the strategy to use for training
        optim(Optimizer): the optimizer to use for training
        loss_fn (callable): the loss function to use for training
        train_dataset (Dataset): the dataset to use for training
        valid_dataset (Dataset): the dataset to use for validation
        eval_dataset (Dataset): the dataset to use for evaluation
        batch_size (int, defaults to 1): the batch size while training
        max_epochs (int, defaults to 2): the number of epochs to train
    """

    def __init__(
        self,
        model,
        strategy: Strategy,
        optim: Optimizer,
        loss_fn,
        train_dataset: Dataset,
        valid_dataset: Dataset,
        eval_dataset: Dataset,
        batch_size: int = 1,
        max_epochs: int = 1,
        accumulation_steps: int = 8,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        train_sampler = None

        if dist.is_initialized() and dist.get_world_size() > 1:
            train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=42, drop_last=True)
        self.train_dataloader = DataLoader(train_dataset,
                                           shuffle=(train_sampler is None),
                                           sampler=train_sampler,
                                           batch_size=batch_size)
        self.valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
        self.eval_dataloader = DataLoader(eval_dataset, batch_size=4*batch_size, shuffle=True)

        self.model = strategy.setup_model(model)
        self.loss_fn = loss_fn
        self.optimizer = strategy.setup_optimizer(optim, self.model)
        self.accumulation_steps = accumulation_steps
        num_update_steps_per_epoch = len(self.train_dataloader) // self.accumulation_steps
        max_steps = math.ceil(self.epochs * num_update_steps_per_epoch)

        self.scheduler = get_scheduler("cosine", self.optimizer, 
                                       num_warmup_steps=0,#math.ceil(max_steps * 0.03),
                                       num_training_steps=max_steps)

    def eval_acc(self, dataloader, logger):
        dist = 0
        on = 0
        cnt = 0
        self.model.eval()
        with torch.no_grad():
            for chosen_ids, c_mask, reject_ids, r_mask in dataloader:
                ids = torch.cat([chosen_ids, reject_ids], dim = 0)
                ids = ids.squeeze(1).to(torch.cuda.current_device())

                masks = torch.cat([c_mask, r_mask], dim = 0)
                masks = masks.squeeze(1).to(torch.cuda.current_device())

                reward = self.model(ids, attention_mask=masks)
                chosen_reward, reject_reward = torch.chunk(reward, 2, dim = 0)

                for i in range(len(chosen_reward)):
                    cnt += 1
                    if chosen_reward[i] > reject_reward[i]:
                        on += 1
                dist += (chosen_reward - reject_reward).mean().item()
            dist_mean = dist / len(dataloader)
            acc = on / cnt
        self.model.train()
        return dist_mean, acc

    def fit(self, logger, log_interval:int=100, save_path:str = "model_rw"):
        time = datetime.now()
        total_loss = 0
        epoch_bar = tqdm(range(self.epochs), desc='Train epoch', disable=not is_rank_0())
        for epoch in range(self.epochs):
            step_bar = tqdm(range(len(self.train_dataloader) // self.accumulation_steps),
                            desc=f'Train step of epoch {epoch + 1}',
                            disable=not is_rank_0())
            # train
            self.model.train()
            cnt = 0
            acc = 0
            dist = 0
            for batch_id, (chosen_ids, c_mask, reject_ids, r_mask) in enumerate(self.train_dataloader):
                chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())
                chosen_reward = self.model(chosen_ids, attention_mask=c_mask)
                reject_reward = self.model(reject_ids, attention_mask=r_mask)
                loss = self.loss_fn(chosen_reward, reject_reward)

                total_loss += loss.item()
                loss = loss / self.accumulation_steps
                self.strategy.backward(loss, self.model, self.optimizer)
                
                # gradient accumulation
                cnt += 1
                if (batch_id + 1) % self.accumulation_steps == 0:
                    self.strategy.optimizer_step(self.optimizer)
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    step_bar.update()

                    if cnt == log_interval * self.accumulation_steps:
                        if is_rank_0():
                            logger.info(f'Train Epoch {epoch+1}/{self.epochs} loss {total_loss/cnt} batch_id {batch_id}')
                        step_bar.set_postfix({'dist': dist, 'acc': acc, "loss":total_loss/cnt})
                        total_loss = 0
                        cnt = 0

            # eval
            dist, acc = self.eval_acc(self.eval_dataloader, logger)
            if is_rank_0():
                log = pd.DataFrame([[step_bar.n, total_loss/cnt, dist, acc]], columns=['step', 'loss', 'dist', 'acc'])
                log.to_csv(f'{save_path}/log_{time}.csv', mode='a', header=False, index=False)
                logger.info(f'Train Epoch {epoch+1}/{self.epochs} loss {total_loss/cnt} acc {acc} batch_id {batch_id}')
            epoch_bar.update()
            step_bar.set_postfix({'dist': dist, 'acc': acc, "loss":total_loss/cnt})
            step_bar.close()

    def save_model(self,
                   path: str,
                   only_rank0: bool = False,
                   tokenizer: Optional[PreTrainedTokenizerBase] = None) -> None:
        self.strategy.save_model(model=self.model, path=path, only_rank0=only_rank0, tokenizer=tokenizer)
