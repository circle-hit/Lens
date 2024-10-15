import os
from dataclasses import field, dataclass
from typing import Optional, Any
import sys

import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig, HfArgumentParser, AutoModelForCausalLM


from transformers import AutoTokenizer
import random
import re

from tqdm import tqdm

import json

from dataclasses import dataclass, field
import logging
logging.basicConfig(level=logging.INFO)

random.seed(112)
import torch
import numpy as np

CURRENT_DIR = os.path.dirname(__file__)

@dataclass
class Arguments:
    """
    Arguments pertaining to what data we are going to input our model for eval.
    """
    model_name_or_path: str = field(
        default=None, metadata={"help": "model_name_or_path"}
    )
    dataset: str = field(
        default=None, metadata= {"help": "which dataset to use"}
    )
    language_name: str = field(
        default=None, metadata={"help": "language"}
    )
    output_dir: str = field(
        default=None, metadata={"help": "output_dir"}
    )
    num: int = field(
        default=1000, metadata={"help": "data number when sampling"}
    )
    overwrite: bool = field(
        default=True
    )
    bs: int = field(
        default=16
    )
    sys_prompt: str = field(
        default=""
    )
    if_support_sys_prompt: bool = field(
        default=True
    )
    split_word: str = field(
        default="assistant"
    )
    indepent_tokenizer: str = field(
        default=""
    )
    if_split_by_special: bool = field(
        default=False
    )  
def batch_generating(model, prompts, tokenizer, generation_config, if_support_sys_prompt, split_word, IDs, language, sys_prompt):
    

    if sys_prompt == "":
        t_sys_prompt = ""
    elif sys_prompt == "phi":
        t_sys_prompt = "You are a helpful assistant."
    messages = [[
            {"role": "system", "content": t_sys_prompt},
            {"role": "user", "content": prompt}
        ] for prompt in prompts]


    inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(inputs, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False, temperature=1.0, top_p=None, generation_config=generation_config, pad_token_id=tokenizer.pad_token_id)

    '''
    bs, dict_size
    ''' 
    
    logits = outputs.scores[0]
    logits = logits.float().cpu().detach()
    
    anss = []
    responses = []
    responses_token = []
    for a_logits in logits:
        choicesAll_logits = a_logits[[IDs[0][0], IDs[0][1], IDs[0][2], IDs[0][3]]].numpy()
        assert not (np.any(np.isinf(choicesAll_logits)) or np.any(np.isnan(choicesAll_logits)))
        if language == "ar":
            ans = {0: 'أ', 1: 'ب', 2: 'ك', 3: 'د'}[np.argmax(choicesAll_logits)]        
        else:
            ans = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(choicesAll_logits)]
        response = tokenizer.decode([a_logits.argmax(-1).item()])
        
        responses_token.append(a_logits.argmax(-1).item())
        responses.append(response)
        anss.append(ans)


        
    return responses, anss, responses_token



from load import load_mydataset
from process import post_process_results
from build import build_prmompts as build_prmompts
from label import extract_labels


def main():
    parser = HfArgumentParser((Arguments))
    args = parser.parse_args_into_dataclasses()[0]

    output_path = os.path.join(args.output_dir, args.dataset + "_" + args.language_name + ".json")

    generate = False
    if not args.overwrite:
        if os.path.exists(output_path):
            print("file exist")
            generate = False
        else:
            generate = True
    else:
        generate = True      

    if generate:
        if args.indepent_tokenizer == "":
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = args.model_name_or_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = args.indepent_tokenizer)  
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"



        if args.language_name == "ar":
            sA_id = tokenizer.encode('أ', add_special_tokens=False)[0]
            sB_id = tokenizer.encode('ب', add_special_tokens=False)[0]
            sC_id = tokenizer.encode('ك', add_special_tokens=False)[0]
            sD_id = tokenizer.encode('د', add_special_tokens=False)[0]

        else:
            sA_id = tokenizer.encode("A", add_special_tokens=False)[0]
            sB_id = tokenizer.encode("B", add_special_tokens=False)[0]
            sC_id = tokenizer.encode("C", add_special_tokens=False)[0]
            sD_id = tokenizer.encode("D", add_special_tokens=False)[0]

        
        IDs = [
            [sA_id, sB_id, sC_id, sD_id]
        ]



        generation_config = GenerationConfig.from_pretrained(args.model_name_or_path)
        generation_config.max_new_tokens = 1
        generation_config.num_beams = 1
        generation_config.output_scores=True
        generation_config.return_dict_in_generate=True

        
        
        model1 = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16, device_map="auto",  use_flash_attention_2 = True)

        dataset = load_mydataset(args.dataset, args.language_name, args.num)


            
        all_built_prompts = build_prmompts(dataset, args.dataset, args.language_name)
        all_extracted_labels = extract_labels(dataset, args.dataset, args.language_name)

        for_dict_prompts = all_built_prompts
        for_dict_answers = []
        for_dict_choices = []
        for_dict_labels = all_extracted_labels
        for_dict_tokens = []


        i = 0
        for i in tqdm(range(0, len(all_built_prompts), args.bs)):

            index1 = i
            index2 = min((i + args.bs), len(all_built_prompts))

            prompts = all_built_prompts[index1: index2]
            answers, choices, responses_token= batch_generating(model1, prompts, tokenizer, generation_config, args.if_support_sys_prompt, args.split_word, IDs, args.language_name, args.sys_prompt)
            
            for_dict_answers.extend(answers)
            for_dict_choices.extend(choices)
            for_dict_tokens.extend(responses_token)
        
        print(for_dict_choices[0:2])
        print(for_dict_labels[0:2])

        correct = 0
        for choice, label in zip(for_dict_choices, for_dict_labels):
            if choice == label:
                correct += 1

        all_index = len(all_built_prompts)
        print(correct/all_index)
        print(round(correct/all_index, 4))
        every_qa = []
        for prompt, answer, choice, label, token in zip(for_dict_prompts, for_dict_answers, for_dict_choices, for_dict_labels, for_dict_tokens):
            temp_dict = {
                "prompt": prompt,
                "token": token, 
                "answer": answer,
                "choice": choice,
                "label": label
            }
            every_qa.append(temp_dict)

        json_data = {
            
            "Correct": correct,
            "All": all_index,
            "Metric": correct/all_index,
            "Instances": every_qa
        }

        with open(output_path, "w") as f:
            json.dump(json_data, f, indent=4, ensure_ascii=False)

        

if __name__ == "__main__":
    main()
