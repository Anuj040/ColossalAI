from typing import Optional

from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from transformers import GPTNeoXJapaneseConfig, GPTNeoXJapaneseForCausalLM
from ..base import Actor
import torch.distributed as dist
import torch


class GPTActor(Actor):
    """
    GPT Actor model.

    Args:
        pretrained (str): Pretrained model name or path.
        config (GPT2Config): Model config.
        checkpoint (bool): Enable gradient checkpointing.
        lora_rank (int): Rank of the LoRa layer.
        lora_train_bias (str): Bias training strategy for the LoRa layer.
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
            model = GPTNeoXJapaneseForCausalLM(GPTNeoXJapaneseConfig.from_pretrained(pretrained))
            if state_dict is None: 
                model = GPTNeoXJapaneseForCausalLM.from_pretrained(pretrained)
                # model = GPT2LMHeadModel.from_pretrained(pretrained)
            elif shard_init:
                for n, p in model.named_parameters():
                    x = state_dict[n]
                    x = x.chunk(torch.cuda.device_count(), dim=-1)
                    x = x[dist.get_rank()]
                    p.data.copy_(x)
            else:
                model.load_state_dict(state_dict)
        elif config is not None:
            model = GPT2LMHeadModel(config)
        else:
            model = GPT2LMHeadModel(GPT2Config())
        if checkpoint:
            model.gradient_checkpointing_enable()
        super().__init__(model, lora_rank, lora_train_bias)
