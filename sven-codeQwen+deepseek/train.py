import os
import torch
import logging
import argparse

from sven.trainer import PrefixTrainer, TextPromptTrainer
from sven.utils import set_seed, set_logging, set_devices
from sven.constant import MODEL_DIRS


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_name', type=str, required=True)

    parser.add_argument('--data_dir', type=str, default='../data_train_val')
    parser.add_argument('--output_dir', type=str, default='../trained/')
    parser.add_argument('--model_type', type=str, default='prefix')
    parser.add_argument('--pretrain_dir', type=str, default=None)
    parser.add_argument('--vul_type', type=str, default=None)

    parser.add_argument('--n_prefix_token', type=int, default=None)
    parser.add_argument('--num_train_epochs', type=int, default=None)
    parser.add_argument('--kl_loss_ratio', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)

    parser.add_argument('--contrastive_loss_ratio', type=int, default=400)
    parser.add_argument('--max_num_tokens', type=int, default=1024)
    parser.add_argument('--grad_acc_steps', type=int, default=2)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--diff_level', type=str, choices=['prog', 'line', 'char', 'mix'], default='mix')
    parser.add_argument('--lm_loss_ratio', type=int, default=1)

    parser.add_argument('--logging_steps', type=int, default=100)
    parser.add_argument('--save_epochs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    if args.pretrain_dir is None:
        args.pretrain_dir = 'Qwen/Qwen2.5-Coder-1.5B-Instruct'

    if args.pretrain_dir in MODEL_DIRS:
        args.pretrain_dir = MODEL_DIRS[args.pretrain_dir]

    # Handle n_prefix_token
    if args.n_prefix_token is None:
        if args.pretrain_dir == 'Salesforce/codegen-350M-multi':
            args.n_prefix_token = 5
        elif args.pretrain_dir == 'Salesforce/codegen-2B-multi':
            args.n_prefix_token = 8
        elif args.pretrain_dir == 'Salesforce/codegen-6B-multi':
            args.n_prefix_token = 12
        elif args.pretrain_dir == 'm-a-p/OpenCodeInterpreter-DS-33B':
            args.n_prefix_token = 10
        elif args.pretrain_dir == 'Qwen/Qwen2.5-Coder-1.5B-Instruct':
            args.n_prefix_token = 7
        elif args.pretrain_dir == 'deepseek-ai/deepseek-coder-1.3b-base':  # ✅ DeepSeek
            args.n_prefix_token = 8
        else:
            assert False, f"Unknown model for setting n_prefix_token: {args.pretrain_dir}"

    # Handle num_train_epochs
    if args.num_train_epochs is None:
        if args.pretrain_dir == 'Salesforce/codegen-350M-multi':
            args.num_train_epochs = 8
        elif args.pretrain_dir == 'Salesforce/codegen-2B-multi':
            args.num_train_epochs = 5
        elif args.pretrain_dir == 'Salesforce/codegen-6B-multi':
            args.num_train_epochs = 5
        elif args.pretrain_dir == 'm-a-p/OpenCodeInterpreter-DS-33B':
            args.num_train_epochs = 6
        elif args.pretrain_dir == 'Qwen/Qwen2.5-Coder-1.5B-Instruct':
            args.num_train_epochs = 5
        elif args.pretrain_dir == 'deepseek-ai/deepseek-coder-1.3b-base':  # ✅ DeepSeek
            args.num_train_epochs = 5
        else:
            assert False, f"Unknown model for setting num_train_epochs: {args.pretrain_dir}"

    # Handle kl_loss_ratio
    if args.kl_loss_ratio is None:
        if args.pretrain_dir == 'Salesforce/codegen-350M-multi':
            args.kl_loss_ratio = 1600
        elif args.pretrain_dir == 'Salesforce/codegen-2B-multi':
            args.kl_loss_ratio = 1600
        elif args.pretrain_dir == 'Salesforce/codegen-6B-multi':
            args.kl_loss_ratio = 2000
        elif args.pretrain_dir == 'm-a-p/OpenCodeInterpreter-DS-33B':
            args.kl_loss_ratio = 1800
        elif args.pretrain_dir == 'Qwen/Qwen2.5-Coder-1.5B-Instruct':
            args.kl_loss_ratio = 1700
        elif args.pretrain_dir == 'deepseek-ai/deepseek-coder-1.3b-base':  # ✅ DeepSeek
            args.kl_loss_ratio = 1600
        else:
            assert False, f"Unknown model for setting kl_loss_ratio: {args.pretrain_dir}"

    # Learning rate logic
    if args.model_type == 'prefix':
        if args.learning_rate is None:
            args.learning_rate = 1e-2
        if args.contrastive_loss_ratio == 0:
            args.learning_rate = 5e-2
            args.grad_acc_steps *= 2
        if args.diff_level in ('prog', 'line'):
            args.learning_rate = 1e-3
    elif args.model_type == 'text':
        args.learning_rate = 5e-5

    args.output_dir = os.path.join(args.output_dir, args.output_name)
    return args

def main():
    args = get_args()
    set_logging(args, os.path.join(args.output_dir, 'train.log'))
    set_devices(args)
    set_seed(args)

    if args.model_type == 'prefix':
        trainer = PrefixTrainer(args)
    elif args.model_type == 'text':
        trainer = TextPromptTrainer(args)
    else:
        raise NotImplementedError()

    trainer.run()

if __name__ == '__main__':
    main()