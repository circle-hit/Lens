import os
import json
import torch
import logging
import numpy as np

from tqdm import tqdm   
from utils import load_from_pickle, save_to_pickle, build_aligner, projection


class Probe:
    def __init__(self, model, tokenizer, cache_path, data_dir, language_list: list, save_tag: str, random_dps: bool=True, pref_data_dps: int=-1, max_input_length: int=64):
        self.model = model
        self.model.eval()

        self.tokenizer = tokenizer
        self.pref_data_dps = pref_data_dps
        self.random_dps = random_dps
        self.data_dir = data_dir
        self.cache_path = cache_path
        self.language_list = language_list
        self.save_tag = save_tag
        self.max_input_length = max_input_length

    def _load_preference_data(self):
        num_dps = self.pref_data_dps
        filepath = os.path.join(self.data_dir, 'test.jsonl')

        if not os.path.exists(filepath):
            logging.error(f'File not found at: {filepath}')
            return

        lang_data = {lang: [] for lang in self.language_list}
        with open(filepath, 'r') as f:
            for line in f:
                for lang in self.language_list:
                    raw_data = json.loads(line)[lang].strip()
                    messages = [
                        {"role": "system", "content": 'You are a helpful assistant.'},
                        {"role": "user", "content": raw_data},
                    ]
                    data = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    lang_data[lang].append(data)
        
        for k, v in lang_data.items():
            print(f"{k}_data_example: \n{v[0]}\n\n")

        if num_dps != -1:  # 4096 points
            if not self.random_dps:
                preferred_data = preferred_data[:num_dps]
                non_preferred_data = non_preferred_data[:num_dps]
            else:
                indices = np.random.choice(len(preferred_data), num_dps, replace=False)
                preferred_data = [preferred_data[i] for i in indices]
                non_preferred_data = [non_preferred_data[i] for i in indices]
        
        for k, v in lang_data.items():
            logging.info(f'Loaded {len(v)} {k} samples. \n')

        lang_data = {k: self.tokenizer(v, return_tensors="pt", padding=True, truncation=True, max_length=self.max_input_length) for k, v in lang_data.items()}
        return lang_data

    def _get_hidden_sentence_embeddings(self, inputs):
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        batch_size = min(50, input_ids.size(0))
        num_batches = input_ids.size(0) // batch_size
        sent_embs = []

        for i in range(num_batches):
            batch_input_ids = input_ids[i * batch_size: (i + 1) * batch_size]
            batch_attention_mask = attention_mask[i * batch_size: (i + 1) * batch_size]
            logging.info(f'Batch {i + 1}/{num_batches} of size {batch_input_ids.size(0)}')

            with torch.no_grad():
                outputs = self.model(input_ids=batch_input_ids.to(self.model.device), attention_mask=batch_attention_mask.to(self.model.device), output_hidden_states=True)
                hidden_states = outputs.hidden_states  # Tuple of len L tensors: (N, seq_len, D), N = batch_size
            del outputs
            hidden_states = hidden_states[1:]  # Remove the input layer embeddings
            hidden_states = torch.stack(hidden_states)  # (L, N, seq_len, D)
            last_layer_emb = hidden_states[-1]
            hidden_states[-1] = last_layer_emb

            # hidden_sent_embs = torch.mean(hidden_states, dim=2)  # (L, N, D)
            hidden_sent_embs = hidden_states[:, :, -1, :]
            sent_embs.append(hidden_sent_embs.detach().to('cpu'))
            del hidden_sent_embs, hidden_states
            torch.cuda.empty_cache()

        # sent_embs is a list of tensors of shape (L, N, D). Concatenate them along the batch dimension
        hidden_sent_embs = torch.cat(sent_embs, dim=1)  # (L, N, D)
        del sent_embs
        logging.info(f'Hidden sent: {hidden_sent_embs.shape}')
        torch.cuda.empty_cache()
        return hidden_sent_embs

    def compute(self):
        lang_data = self._load_preference_data()
        source_lan_emb = {}

        for lang, data in tqdm(lang_data.items(), desc='Computing sentence embeddings'):
            sent_embs = load_from_pickle(os.path.join(self.cache_path, self.save_tag, f'{lang}_hidden_last.pkl'))
            if sent_embs is None:
                sent_embs = self._get_hidden_sentence_embeddings(data)  # (L, N, D)
                save_to_pickle(sent_embs, os.path.join(self.cache_path, self.save_tag, f'{lang}_hidden_last.pkl'))
            source_lan_emb[lang] = sent_embs
        
        rank = len(self.language_list) - 1
        preference_matrix, Wu_matrix = [], []
        for layer_num in tqdm(range(source_lan_emb["en"].shape[0]), desc='Computing preference matrix'):
            cur_source_lan_emb = {lang: emb[layer_num].numpy() for lang, emb in source_lan_emb.items()}
            Wu, aligner = build_aligner(rank, cur_source_lan_emb)
            preference_matrix.append(torch.tensor(aligner.T))
            Wu_matrix.append(torch.tensor(Wu.T))

        preference_matrix = torch.stack(preference_matrix, dim=0)
        Wu_matrix = torch.stack(Wu_matrix, dim=0)
        logging.info('Preference matrix calculated.')

        save_to_pickle(preference_matrix, os.path.join(self.cache_path, self.save_tag, 'lang_specific_space_last.pkl'))
        save_to_pickle(Wu_matrix, os.path.join(self.cache_path, self.save_tag, 'lang_shared_space_last.pkl'))
        
        lang_specific_proj = preference_matrix

        en_direction = []
        for layer_idx in range(len(source_lan_emb['en'])):
            en_direction.append(projection(source_lan_emb['en'][layer_idx], lang_specific_proj[layer_idx].to(source_lan_emb['en'].dtype).to(source_lan_emb['en'].device)))

        for lang in source_lan_emb.keys():
            lang_direction = []
            for layer_idx in range(len(source_lan_emb[lang])):
                lang_direction.append(projection(source_lan_emb[lang][layer_idx], lang_specific_proj[layer_idx].to(source_lan_emb[lang].dtype).to(source_lan_emb[lang].device)))

            delta = [lang.mean(dim=0) - en.mean(dim=0) for lang, en in zip(lang_direction, en_direction)]

            save_to_pickle(delta, os.path.join(self.cache_path, self.save_tag, f'{lang}_direction_last.pkl'))
