import os
import sys
import inspect
import argparse
import numpy as np
from probe import Probe
from utils import load_large_model, load_config, seed_all

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='Probing/configs/single/llama3-instruct-8b.ini', type=str, help='Config Path')

args = parser.parse_args()
config = load_config(args.config)

language_list = config.get('Common', 'language_list').split(',')
save_tag = config.get('Common', 'save_tag')
seed = config.getint('Common', 'seed')
seed_all(seed)

max_input_length = config.getint('Dataset', 'max_input_length')
pref_data_dps = config.getint('Dataset', 'pref_data_dps')
random_dps = config.getboolean('Dataset', 'random_dps')
dataset_dir = config.get('Dataset', 'dataset_dir')
dataset_name = config.get('Dataset', 'dataset_name')
data_dir = os.path.join(dataset_dir, dataset_name)

model_id = config.get('Model', 'model_name')
model, tokenizer = load_large_model(model_id)

# Language Space Probing
probe = Probe(model=model, tokenizer=tokenizer,
                cache_path=f"Probing/language_space/{model_id.split('-')[0]}",
                data_dir=data_dir,
                language_list=language_list,
                save_tag=save_tag,
                random_dps=random_dps,
                pref_data_dps=pref_data_dps,
                max_input_length=max_input_length)
probe.compute()