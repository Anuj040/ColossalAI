import argparse
import os
import pandas as pd
import torch
import torch.distributed as dist
from coati.dataset import DataCollatorForSupervisedDataset, PromptDataset, SupervisedDataset
from coati.models.bloom import BLOOMRM, BLOOMActor, BLOOMCritic
from coati.models.gpt import GPTRM, GPTActor, GPTCritic
from coati.models.llama import LlamaActor, LlamaCritic, LlamaRM
from coati.models.opt import OPTRM, OPTActor, OPTCritic
from coati.models.roberta import RoBERTaRM, RoBERTaActor, RoBERTaCritic
from coati.trainer import PPOTrainer
from coati.trainer.strategies import ColossalAIStrategy, DDPStrategy, NaiveStrategy,  Strategy
from coati.utils import prepare_llama_tokenizer_and_embedding
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, BloomTokenizerFast, GPT2Tokenizer, LlamaTokenizer, RobertaTokenizer, GPTNeoXJapaneseTokenizer
from transformers import AutoModelForCausalLM, AutoModel, GPTNeoXJapaneseForCausalLM, GPTNeoXJapaneseModel
import torch.distributed as dist

from colossalai.nn.optimizer import HybridAdam
from torch import nn
def is_rank_0() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0

def get_model_numel(model: nn.Module, strategy: Strategy) -> int:
    numel = sum(p.numel() for p in model.parameters())
    if isinstance(strategy, ColossalAIStrategy) and strategy.stage == 3 and strategy.shard_init:
        numel *= dist.get_world_size()
    return numel


def print_model_numel(model_dict: dict) -> None:
    B = 1024**3
    M = 1024**2
    K = 1024
    outputs = ''
    for name, numel in model_dict.items():
        outputs += f'{name}: '
        if numel >= B:
            outputs += f'{numel / B:.2f} B\n'
        elif numel >= M:
            outputs += f'{numel / M:.2f} M\n'
        elif numel >= K:
            outputs += f'{numel / K:.2f} K\n'
        else:
            outputs += f'{numel}\n'

    if is_rank_0():
        print(outputs)


def main(args):
    # configure strategy
    if args.strategy == 'naive':
        strategy = NaiveStrategy()
    elif args.strategy == 'ddp':
        strategy = DDPStrategy()
    elif args.strategy == 'colossalai_gemini':
        strategy = ColossalAIStrategy(stage=3, placement_policy='cuda', initial_scale=2**5, shard_init=True)
    elif args.strategy == 'colossalai_zero2':
        strategy = ColossalAIStrategy(stage=2, placement_policy='cuda')
    elif args.strategy == "colossalai_zero2_cpu":
        strategy = ColossalAIStrategy(stage=2, placement_policy='cpu')
    elif args.strategy == 'colossalai_zero1':
        strategy = ColossalAIStrategy(stage=1, placement_policy='cuda')
    elif args.strategy == 'colossalai_zero1_cpu':
        strategy = ColossalAIStrategy(stage=3, placement_policy='cpu', shard_init=True)
    else:
        raise ValueError(f'Unsupported strategy "{args.strategy}"')

    if args.rm_path is not None:
        if not args.rm_path.startswith("/"):
            args.rm_path = os.path.abspath(args.rm_path)
        weight_file_names = [file_name for file_name in os.listdir(args.rm_path) if file_name.startswith("pytorch") and file_name.endswith(".bin")]
        stata_dict_files = [os.path.join(args.rm_path, file_name) for file_name in weight_file_names]

        # Load weights
        for i, state_dict_file in enumerate(stata_dict_files):
            loaded_state_dict = torch.load(state_dict_file, map_location='cpu')
            if i == 0:
                reward_state_dict = loaded_state_dict
            else:
                reward_state_dict.update(loaded_state_dict)
    else:
        reward_state_dict = GPTNeoXJapaneseModel.from_pretrained(args.rm_pretrain, low_cpu_mem_usage=True).to("cpu").state_dict()

    if args.model_path is not None:
        if not args.model_path.startswith("/"):
            args.model_path = os.path.abspath(args.model_path)
        weight_file_names = [file_name for file_name in os.listdir(args.model_path) if file_name.startswith("pytorch") and file_name.endswith(".bin")]
        stata_dict_files = [os.path.join(args.model_path, file_name) for file_name in weight_file_names]

        # Load weights
        for i, state_dict_file in enumerate(stata_dict_files):
            loaded_state_dict = torch.load(state_dict_file, map_location='cpu')
            if i == 0:
                actor_state_dict = loaded_state_dict
            else:
                actor_state_dict.update(loaded_state_dict)
    else:
        actor_state_dict = GPTNeoXJapaneseForCausalLM.from_pretrained(args.pretrain, low_cpu_mem_usage=True).to("cpu").state_dict()

    # configure model
    with strategy.model_init_context():
        if args.model == 'gpt2':
            initial_model = GPTActor(pretrained=args.pretrain,lora_rank=args.lora_rank, state_dict = actor_state_dict)
        elif args.model == 'bloom':
            initial_model = BLOOMActor(pretrained=args.pretrain,lora_rank=args.lora_rank)
        elif args.model == 'opt':
            initial_model = OPTActor(pretrained=args.pretrain,lora_rank=args.lora_rank)
        elif args.model == 'llama':
            initial_model = LlamaActor(pretrained=args.pretrain,lora_rank=args.lora_rank)
        elif args.model == 'roberta':
            initial_model = RoBERTaActor(pretrained=args.pretrain,lora_rank=args.lora_rank)
        else:
            raise ValueError(f'Unsupported actor model "{args.model}"')

        if args.strategy != 'colossalai_gemini':
            initial_model.to(torch.float16).to(torch.cuda.current_device())

        if args.model == 'gpt2':
            actor = GPTActor(pretrained=args.pretrain, lora_rank=args.lora_rank, state_dict = actor_state_dict)
        elif args.model == 'bloom':
            actor = BLOOMActor(pretrained=args.pretrain, lora_rank=args.lora_rank)
        elif args.model == 'opt':
            actor = OPTActor(pretrained=args.pretrain, lora_rank=args.lora_rank)
        elif args.model == 'llama':
            actor = LlamaActor(pretrained=args.pretrain, lora_rank=args.lora_rank)
        elif args.model == 'roberta':
            actor = RoBERTaActor(pretrained=args.pretrain, lora_rank=args.lora_rank)
        else:
            raise ValueError(f'Unsupported actor model "{args.model}"')
        
        if args.strategy != 'colossalai_gemini':
            actor.to(torch.float16).to(torch.cuda.current_device())

        del actor_state_dict
  
        rm_model_name = args.model if args.rm_model is None else args.rm_model
        if rm_model_name == 'gpt2':
            reward_model = GPTRM(pretrained=args.rm_pretrain,lora_rank=args.lora_rank, state_dict = reward_state_dict)
        elif rm_model_name == 'bloom':
            reward_model = BLOOMRM(pretrained=args.rm_pretrain,lora_rank=args.lora_rank)
        elif rm_model_name == 'opt':
            reward_model = OPTRM(pretrained=args.rm_pretrain,lora_rank=args.lora_rank)
        elif rm_model_name == 'llama':
            reward_model = LlamaRM(pretrained=args.rm_pretrain,lora_rank=args.lora_rank)
        elif rm_model_name == 'roberta':
            reward_model = RoBERTaRM(pretrained=args.rm_pretrain,lora_rank=args.lora_rank)
        else:
            raise ValueError(f'Unsupported reward model "{rm_model_name}"')
        
        if args.strategy != 'colossalai_gemini':
            reward_model.to(torch.float16).to(torch.cuda.current_device())

        if rm_model_name == 'gpt2':
            critic = GPTCritic(pretrained=args.rm_pretrain, lora_rank=args.lora_rank, use_action_mask=True, state_dict = reward_state_dict)
        elif rm_model_name == 'bloom':
            critic = BLOOMCritic(pretrained=args.rm_pretrain, lora_rank=args.lora_rank, use_action_mask=True)
        elif rm_model_name == 'opt':
            critic = OPTCritic(pretrained=args.rm_pretrain, lora_rank=args.lora_rank, use_action_mask=True)
        elif rm_model_name == 'llama':
            critic = LlamaCritic(pretrained=args.rm_pretrain, lora_rank=args.lora_rank, use_action_mask=True)
        elif rm_model_name == 'roberta':
            critic = RoBERTaCritic(pretrained=args.rm_pretrain, lora_rank=args.lora_rank, use_action_mask=True)
        else:
            raise ValueError(f'Unsupported reward model "{rm_model_name}"')

        del reward_state_dict

        if args.strategy != 'colossalai_gemini':
            critic.to(torch.float16).to(torch.cuda.current_device())

    actor_numel = get_model_numel(actor, strategy)
    critic_numel = get_model_numel(critic, strategy)
    initial_model_numel = get_model_numel(initial_model, strategy)
    reward_model_numel = get_model_numel(reward_model, strategy)
    print_model_numel({
        'Actor': actor_numel,
        'Critic': critic_numel,
        'Initial model': initial_model_numel,
        'Reward model': reward_model_numel
    })

    # configure optimizer
    if args.strategy.startswith('colossalai'):
        actor_optim = HybridAdam(actor.parameters(), lr=1e-7)
        critic_optim = HybridAdam(critic.parameters(), lr=1e-7)
    else:
        actor_optim = Adam(actor.parameters(), lr=1e-7)
        critic_optim = Adam(critic.parameters(), lr=1e-7)

    # configure tokenizer
    if args.model == 'gpt2':
        # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        # tokenizer.pad_token = tokenizer.eos_token
        tokenizer = GPTNeoXJapaneseTokenizer.from_pretrained(args.pretrain)
    elif args.model == 'bloom':
        tokenizer = BloomTokenizerFast.from_pretrained('bigscience/bloom-560m')
    elif args.model == 'opt':
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
    elif args.model == 'llama':
        tokenizer = LlamaTokenizer.from_pretrained(args.pretrain)
        tokenizer.eos_token = '<\s>'
    elif args.model == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    else:
        raise ValueError(f'Unsupported model "{args.model}"')

    if args.model == 'llama':
        tokenizer = prepare_llama_tokenizer_and_embedding(tokenizer, actor)
    else:
        tokenizer.pad_token = tokenizer.eos_token

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    prompt_dataset = PromptDataset(tokenizer=tokenizer, data_path=args.prompt_path,  max_datasets_size=40)
    if dist.is_initialized() and dist.get_world_size() > 1:
        prompt_sampler = DistributedSampler(prompt_dataset, shuffle=True, seed=42, drop_last=True)
    prompt_dataloader = DataLoader(prompt_dataset,
                                   shuffle=(prompt_sampler is None),
                                   sampler=prompt_sampler,
                                   batch_size=args.train_batch_size)

    pretrain_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=args.pretrain_dataset, max_datasets_size=40)
    if dist.is_initialized() and dist.get_world_size() > 1:
        pretrain_sampler = DistributedSampler(pretrain_dataset, shuffle=True, seed=42, drop_last=True)
    pretrain_dataloader = DataLoader(pretrain_dataset,
                                     shuffle=(pretrain_sampler is None),
                                     sampler=pretrain_sampler,
                                     batch_size=args.ptx_batch_size,
                                     collate_fn=data_collator)

    def tokenize_fn(texts):
        # MUST padding to max length to ensure inputs of all ranks have the same length
        # Different length may lead to hang when using gemini, as different generation steps
        batch = tokenizer(texts, return_tensors='pt', max_length=96, padding='max_length', truncation=True)
        return {k: v.to(torch.cuda.current_device()) for k, v in batch.items()}

    (actor, actor_optim), (critic, critic_optim) = strategy.prepare((actor, actor_optim), (critic, critic_optim))
    
    # configure trainer
    trainer = PPOTrainer(
        strategy,
        actor,
        critic,
        reward_model,
        initial_model,
        actor_optim,
        critic_optim,
        kl_coef=args.kl_coef,
        ptx_coef=args.ptx_coef,
        max_epochs=args.max_epochs,
        train_batch_size=args.train_batch_size,
        experience_batch_size=args.experience_batch_size,
        tokenizer=tokenize_fn,
        max_length=128,
        do_sample=True,
        temperature=1.0,
        top_k=50,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if is_rank_0():
        print("16")

    trainer.fit(prompt_dataloader=prompt_dataloader,
                pretrain_dataloader=pretrain_dataloader,
                num_episodes=args.num_episodes,
                max_timesteps=args.max_timesteps,
                update_timesteps=args.update_timesteps)
    if is_rank_0():
        print("17")

    # save model checkpoint after fitting
    trainer.save_model(args.save_path, only_rank0=True, tokenizer=tokenizer)
    # save optimizer checkpoint on all ranks
    if args.need_optim_ckpt:
        strategy.save_optimizer(actor_optim,
                                os.path.join(args.save_path,f'actor_optim_checkpoint_prompts_{torch.cuda.current_device()}'),
                                only_rank0=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_path', type=str, default="databricks-dolly-15k-ja.json", help='path to the prompt dataset')
    parser.add_argument('--pretrain_dataset', type=str, default="databricks-dolly-15k-ja.json", help='path to the pretrained dataset')
    parser.add_argument('--strategy',
                        choices=['naive', 'ddp', 'colossalai_gemini', 'colossalai_zero2', "colossalai_zero2_cpu", "colossalai_zero1", "colossalai_zero1_cpu"],
                        default='colossalai_gemini',
                        help='strategy to use')
    parser.add_argument('--model', choices=['gpt2', 'bloom', 'opt', 'llama'], default='gpt2')
    parser.add_argument('--pretrain', type=str, default="abeja/gpt-neox-japanese-6.7b")
    parser.add_argument('--model_path', type=str, default='model_sft_6')
    parser.add_argument('--rm_model', default='gpt2', choices=['gpt2', 'bloom', 'opt', 'llama', 'roberta'])
    parser.add_argument('--rm_path', type=str, default='model_rw_4')
    parser.add_argument('--rm_pretrain', type=str, default="abeja/gpt-neox-japanese-6.7b")
    parser.add_argument('--save_path', type=str, default='model_rlhf_1')
    parser.add_argument('--need_optim_ckpt', type=bool, default=True)
    parser.add_argument('--num_episodes', type=int, default=10)
    parser.add_argument('--max_timesteps', type=int, default=10)
    parser.add_argument('--update_timesteps', type=int, default=10)
    parser.add_argument('--max_epochs', type=int, default=5)
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--ptx_batch_size', type=int, default=1)
    parser.add_argument('--experience_batch_size', type=int, default=4)
    parser.add_argument('--lora_rank', type=int, default=1, help="low-rank adaptation matrices rank")
    parser.add_argument('--kl_coef', type=float, default=0.1)
    parser.add_argument('--ptx_coef', type=float, default=0.9)
    args = parser.parse_args()
    main(args)
