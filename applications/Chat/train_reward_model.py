import argparse
from random import randint
import os
import loralib as lora
import torch
from coati.dataset import HhRlhfDataset, RmStaticDataset
from coati.models import LogExpLoss, LogSigLoss
from coati.models.bloom import BLOOMRM
from coati.models.deberta import DebertaRM
from coati.models.gpt import GPTRM
from coati.models.llama import LlamaRM
from coati.models.opt import OPTRM
from coati.models.roberta import RoBERTaRM
from coati.trainer import RewardModelTrainer
from coati.trainer.strategies import ColossalAIStrategy, DDPStrategy, NaiveStrategy
from coati.utils import prepare_llama_tokenizer_and_embedding
from datasets import load_dataset
from torch.optim import Adam
from transformers import AutoTokenizer, BloomTokenizerFast, DebertaV2Tokenizer, LlamaTokenizer, RobertaTokenizer, GPTNeoXJapaneseTokenizer

from colossalai.logging import get_dist_logger
from colossalai.nn.optimizer import HybridAdam


def train(args):
    # configure strategy
    if args.strategy == 'naive':
        strategy = NaiveStrategy()
    elif args.strategy == 'ddp':
        strategy = DDPStrategy()
    elif args.strategy == 'colossalai_gemini':
        strategy = ColossalAIStrategy(stage=3, placement_policy='cuda')
    elif args.strategy == 'colossalai_zero2':
        strategy = ColossalAIStrategy(stage=2, placement_policy='cuda')
    elif args.strategy == 'colossalai_zero2_cpu':
        strategy = ColossalAIStrategy(stage=2, placement_policy='cpu')
    else:
        raise ValueError(f'Unsupported strategy "{args.strategy}"')

    # configure model
    with strategy.model_init_context():
        if args.model == 'bloom':
            model = BLOOMRM(pretrained=args.pretrain, lora_rank=args.lora_rank).to(torch.cuda.current_device())
        elif args.model == 'opt':
            model = OPTRM(pretrained=args.pretrain, lora_rank=args.lora_rank).to(torch.cuda.current_device())
        elif args.model == 'gpt2':
            model = GPTRM(pretrained=args.pretrain, lora_rank=args.lora_rank).to(torch.cuda.current_device())
        elif args.model == 'deberta':
            model = DebertaRM(pretrained=args.pretrain, lora_rank=args.lora_rank).to(torch.cuda.current_device())
        elif args.model == 'llama':
            model = LlamaRM(pretrained=args.pretrain, lora_rank=args.lora_rank).to(torch.cuda.current_device())
        elif args.model == 'roberta':
            model = RoBERTaRM(pretrained=args.pretrain, lora_rank=args.lora_rank).to(torch.cuda.current_device())
        else:
            raise ValueError(f'Unsupported model "{args.model}"')

        if args.model_path is not None:
            state_dict = torch.load(args.model_path)
            model.load_state_dict(state_dict)

    model = model.to(torch.float16)

    # configure tokenizer
    if args.model == 'gpt2':
        # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        # tokenizer.pad_token = tokenizer.eos_token
        tokenizer = GPTNeoXJapaneseTokenizer.from_pretrained(args.pretrain)
    elif args.model == 'bloom':
        tokenizer = BloomTokenizerFast.from_pretrained('bigscience/bloom-560m')
    elif args.model == 'opt':
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
    elif args.model == 'deberta':
        tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v3-large')
    elif args.model == 'llama':
        tokenizer = LlamaTokenizer.from_pretrained(args.pretrain)
    elif args.model == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    else:
        raise ValueError(f'Unsupported model "{args.model}"')
    max_len = args.max_len

    if args.model == 'llama':
        tokenizer = prepare_llama_tokenizer_and_embedding(tokenizer, model)
    else:
        tokenizer.pad_token = tokenizer.eos_token

    # configure optimizer
    if args.strategy.startswith('colossalai'):
        optim = HybridAdam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optim = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    logger = get_dist_logger()

    # configure loss function
    if args.loss_fn == 'log_sig':
        loss_fn = LogSigLoss()
    elif args.loss_fn == 'log_exp':
        loss_fn = LogExpLoss()
    else:
        raise ValueError(f'Unsupported loss function "{args.loss_fn}"')

    # prepare for data and dataset
    if args.subset is not None:
        data = load_dataset(args.dataset, data_dir=args.subset)
    elif os.path.exists(args.dataset):
        data_files = {"train": os.path.join(args.dataset, "rm_miracl_train.json"),
                      "test": os.path.join(args.dataset, "rm_miracl_test.json")}
        extension = "json"
        data = load_dataset(
            extension,
            data_files=data_files,
            cache_dir="/tmp",
            use_auth_token= None,
        )
    else:
        data = load_dataset(args.dataset)

    if args.test:
        train_data = data['train'].select(range(10))
        eval_data = data['test'].select(range(10))
    else:
        train_data = data['train']
        eval_data = data['test']
    valid_data = data['test'].select((randint(0, len(eval_data) - 1) for _ in range(len(eval_data) // 5)))

    if args.dataset == 'Dahoas/rm-static' or os.path.exists(args.dataset):
        train_dataset = RmStaticDataset(train_data, tokenizer, max_len)
        valid_dataset = RmStaticDataset(valid_data, tokenizer, max_len)
        eval_dataset = RmStaticDataset(eval_data, tokenizer, max_len)
    elif args.dataset == 'Anthropic/hh-rlhf':
        train_dataset = HhRlhfDataset(train_data, tokenizer, max_len)
        valid_dataset = HhRlhfDataset(valid_data, tokenizer, max_len)
        eval_dataset = HhRlhfDataset(eval_data, tokenizer, max_len)
    else:
        raise ValueError(f'Unsupported dataset "{args.dataset}"')

    trainer = RewardModelTrainer(model=model,
                                 strategy=strategy,
                                 optim=optim,
                                 loss_fn=loss_fn,
                                 train_dataset=train_dataset,
                                 valid_dataset=valid_dataset,
                                 eval_dataset=eval_dataset,
                                 batch_size=args.batch_size,
                                 max_epochs=args.max_epochs,
                                accumulation_steps=args.accumulation_steps)

    trainer.fit(logger=logger, log_interval=args.log_interval, save_path = args.save_path)
    # save model checkpoint after fitting on only rank0
    trainer.save_model(path=args.save_path, only_rank0=True, tokenizer=tokenizer)
    # save optimizer checkpoint on all ranks
    if args.need_optim_ckpt:
        strategy.save_optimizer(trainer.optimizer,
                                os.path.join(args.save_path,f'rm_optim_checkpoint_{torch.cuda.current_device()}.pt'),
                                only_rank0=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy',
                        choices=['naive', 'ddp', 'colossalai_gemini', 'colossalai_zero2_cpu'],
                        default='colossalai_zero2_cpu')
    parser.add_argument('--model', choices=['gpt2', 'bloom', 'opt', 'llama'], default='gpt2')
    parser.add_argument('--pretrain', type=str, default="abeja/gpt-neox-japanese-6.7b")
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--need_optim_ckpt', type=bool, default=True)
    parser.add_argument('--dataset',
                        type=str,
                        choices=['Anthropic/hh-rlhf', 'Dahoas/rm-static', "rm-static"],
                        default='./rm-static')
    parser.add_argument('--subset', type=str, default=None)
    parser.add_argument('--save_path', type=str, default='model_rw_3')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--lora_rank', type=int, default=1, help="low-rank adaptation matrices rank")
    parser.add_argument('--loss_fn', type=str, default='log_exp', choices=['log_sig', 'log_exp'])
    parser.add_argument('--log_interval', type=int, default=100, help="how many steps to log")
    parser.add_argument('--test', type=bool, default=False)
    parser.add_argument('--accumulation_steps', type=int, default=8)
    args = parser.parse_args()
    train(args)
