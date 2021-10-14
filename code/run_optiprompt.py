import json
import argparse
import os
import random
import sys
import logging

import torch

from run import *
from utils import *

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # File path
    parser.add_argument('--model_name', type=str, default='bert-base-cased', help='the huggingface model name')
    parser.add_argument('--model_dir', type=str, default=None, help='the model directory (if not using --model_name)')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='the output directory to store trained model and prediction results')
    parser.add_argument('--relation_profile', type=str, default=None,
                        help='meta infomation of 41 relations, containing the pre-defined templates')
    # Basic parameters
    parser.add_argument('--do_eval', action='store_true', help="whether to run evaluation")
    parser.add_argument('--do_train', action='store_true', help="whether to run training process")
    parser.add_argument('--debug', action='store_true', help="Use a subset of data for debugging")
    parser.add_argument('--train_data', type=str, default=None)
    parser.add_argument('--dev_data', type=str, default=None)
    parser.add_argument('--test_data', type=str, default=None)
    parser.add_argument("--do_lowercase", action='store_true')
    # Preprocessing/decoding-related parameters
    parser.add_argument('--max_input_length', type=int, default=512)
    parser.add_argument('--max_output_length', type=int, default=100)
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument("--append_another_bos", action='store_true', default=False)
    # Training-related parameters
    parser.add_argument('--train_batch_size', type=int, default=8, help='training batch size per GPU')
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--num_epoch', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=3e-3)
    parser.add_argument('--warmup_proportion', type=float, default=0.1)
    parser.add_argument('--check_step', type=int, default=-1, help='how often to output training loss')
    parser.add_argument('--eval_per_epoch', type=int, default=3)
    parser.add_argument('--wait_step', type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument('--seed', type=int, default=6)
    # optiprompt parameters
    parser.add_argument('--relation', type=str, required=True, help='which relation is considered in this run')
    parser.add_argument('--init_manual_template', action='store_true',
                        help='whether to use manual template to initialize the dense vectors')
    parser.add_argument('--random_init', type=str, default='none', choices=['none', 'embedding', 'all'],
                        help='none: use pre-trained model; embedding: random initialize the embedding layer of the model; all: random initialize the whole model')
    parser.add_argument('--num_vectors', type=int, default=5, help='how many dense vectors are used in OptiPrompt')
    parser.add_argument('--freeze', action='store_true', help='whether to update model parameter')
    args = parser.parse_args()

    if args.do_train:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "train.log"), 'w'))
    else:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "eval.log"), 'w'))

    logger.info(args)
    args.n_gpu = torch.cuda.device_count()
    logger.info('# GPUs: %d' % args.n_gpu)
    if args.n_gpu == 0:
        logger.warning('No GPUs found!')

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    if torch.cuda.device_count() > 1:
        torch.cuda.manual_seed_all(args.seed)

    run(args, logger)