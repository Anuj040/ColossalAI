from typing import Optional

import torch.nn as nn
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import AutoModel, GPTNeoXJapaneseConfig, GPTNeoXJapaneseModel
from ..base import RewardModel
import torch.distributed as dist
import torch

class GPTRM(RewardModel):
    """
    GPT Reward model.

    Args:
        pretrained (str): Pretrained model name or path.
        config (GPT2Config): Model config.
        checkpoint (bool): Enable gradient checkpointing.
        lora_rank (int): Rank of the low-rank approximation.
        lora_train_bias (str): LoRA bias training mode.
    """

    def __init__(self,
                 pretrained: Optional[str] = None,
                 config: Optional[GPT2Config] = None,
                 checkpoint: bool = False,
                 lora_rank: int = 0,
                 lora_train_bias: str = 'none',
                 state_dict:dict = None,
                 shard_init:bool = False) -> None:
        if pretrained is not None:
            model = GPTNeoXJapaneseModel(GPTNeoXJapaneseConfig.from_pretrained(pretrained))
            if state_dict is None: 
                model = GPTNeoXJapaneseModel.from_pretrained(pretrained)
                # model = GPT2Model.from_pretrained(pretrained)
            elif shard_init:
                for n, p in model.named_parameters():
                    x = state_dict[n]
                    x = x.chunk(torch.cuda.device_count(), dim=-1)
                    x = x[dist.get_rank()]
                    p.data.copy_(x)
            else:
                model.load_state_dict(state_dict)
        elif config is not None:
            model = GPT2Model(config)
        else:
            model = GPT2Model(GPT2Config())
        if checkpoint:
            model.gradient_checkpointing_enable()

        value_head = nn.Linear(model.config.hidden_size, 1)
        value_head.weight.data.normal_(mean=0.0, std=1 / (model.config.hidden_size + 1))
        super().__init__(model, value_head, lora_rank, lora_train_bias)
