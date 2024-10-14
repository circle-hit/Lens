import os
import torch
import logging
import pickle
import numpy as np
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
import random
import configparser

logging.getLogger().setLevel(logging.INFO)


MODEL_IDENITFIER = {
    'llama3-instruct-8b': '',
    'llama3.1-instruct-8b': '',
    'phi3.5-mini': '',
}

def build_aligner(rank: int, lan_emb):
    lan_mean_emb = {lan: np.mean(emb, axis=0) for lan, emb in lan_emb.items()}
    W = np.stack(list(lan_mean_emb.values())).T

    _, D = W.shape

    wc = W @ np.ones(D) / D
    u, s, vh = np.linalg.svd(W - wc.reshape(-1, 1) @ np.ones((1, D)))
    Ws, Gamma  = u[:, :rank], vh.T[:, :rank] @ np.diag(s[:rank])
    best_fit_W = wc.reshape(-1, 1) @ np.ones((1, D)) + Ws @ Gamma.T

    wc_new = np.linalg.pinv(best_fit_W).T @ np.ones(D)
    wc_new /= (wc_new ** 2).sum()
    prod = best_fit_W - wc_new.reshape(-1, 1) @ np.ones((1, D))

    u, s, vh = np.linalg.svd(prod)
    ws_new = u[:, :rank]

    return wc_new, ws_new

def projection(emb, direction):
    direction = direction / torch.linalg.norm(direction, axis=1, keepdims=True)
    proj = torch.matmul(torch.matmul(emb, direction.T), direction)

    return proj

def load_large_model(model_id, quantize=False, add_peft=False):
    """
    Load a language model from HuggingFace.
    :param model_id: Name of the model from the MODEL_IDENTIFIER dictionary E.g. 'gpt2', 'mistral', 'zephyr-sft', 'gptj', 'opt'
    :param quantize: If True, quantize the model to 4-bit
    :param add_peft: If True, add LoRA with rank 64 to the model
    :param hf_token: Token for HuggingFace model hub. Required to access Mistral models.
    :return:
    """

    model_path = MODEL_IDENITFIER[model_id]
    tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=512, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if quantize:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        quantization_config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,  # Non quantized weights are torch.float16 by default
        quantization_config=quantization_config,
        trust_remote_code=True,
    )

    if add_peft:
        model = prepare_model_for_kbit_training(model)  # preprocess the quantized model for training
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)

    model.max_length = tokenizer.model_max_length
    model.eval()

    logging.info(f'Model {model_id} loaded.')
    return model, tokenizer

def load_from_pickle(file_path):
    if not os.path.exists(file_path):
        return None
    logging.info(f"Loading hidden states from {file_path}")
    with open(file_path, 'rb') as f:
        return pickle.load(f)
    
def save_to_pickle(data, file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    logging.info(f"Saving hidden states to {file_path}")
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_config(config_path):
    config = configparser.ConfigParser()
    assert os.path.exists(config_path), f'Config file not found at {config_path}'
    config.read(config_path)
    return config

def seed_all(seed):
    set_seed(seed)            # Huggingface
    random.seed(seed)         # Python
    np.random.seed(seed)      # Numpy
    torch.manual_seed(seed)   # PyTorch