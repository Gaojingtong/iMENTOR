import re
import os
from datasets import Dataset, load_dataset
from random import randint, seed, choice
from typing import List, Tuple
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/aime')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='qwen-instruct')

    args = parser.parse_args()

    data_source = 'aime'

    train_dataset = load_dataset('open-r1/DAPO-Math-17k-Processed', 'all', split='train')
    test_dataset = load_dataset('BytedTsinghua-SIA/AIME-2024', split='train')

    def make_map_fn(fn):
        if fn=='train':
            def process_fn(example, idx):
                question = example['source_prompt']
                solution = example['reward_model']
                data = {
                    "data_source": data_source,
                    "prompt": question,
                    "reward_model": solution,
                }
                return data
            return process_fn
        if fn=='test':
            def process_fn(example, idx):
                question = example['prompt']
                solution = example['reward_model']
                ind = example['extra_info']['index']
                data = {
                    "data_source": data_source,
                    "prompt": question,
                    "reward_model": solution,
                    "ind": ind,
                }
                return data
            return process_fn
        raise ValueError("fn:{}, 1:{}, 2:{}".format(fn, fn=='train', fn=='test'))
    
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True).select_columns(['prompt', 'reward_model', 'data_source'])
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True).select_columns(['prompt', 'reward_model', 'data_source', 'ind'])

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir) 
