import logging

import torch
from transformers.data.data_collator import *


logger = logging.getLogger(__name__)

SUPPORTED_DECODER_MODELS = ['codegen', 'bloomz', 'gpt-neox', 'llama', 'vicuna']
SUPPORTED_SEQ2SEQ_MODELS = ['t5', 'flan-t5']

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

def check_model(model_name, supported_models):
    for sup_model in supported_models:
        if sup_model.lower() in model_name.lower():
            return True

    return False

def replace_sublist(lst, sublist, replacement):
    n = len(lst)
    m = len(sublist)
    
    for i in range(n - m + 1):
        if lst[i:i+m] == sublist:
            return lst[:i] + replacement + lst[i+m:]
    
    return lst

@dataclass
class DataCollator:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_source_length: Optional[int] = None
    max_target_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    add_task_name: bool = False
    add_dataset_name: bool = False
    common_dataset_name: str = None
    text_only: bool = False
    num_examples: int = 0
    input_record_file: str = None
    lang_list: list = None

    def __call__(self, batch, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        model_name = self.model.config._name_or_path
        self.tokenizer.padding_side = 'left'
        model_inputs = {}
        
        for lang in self.lang_list:
            input_ids, attention_mask, labels = [], [], []
            for instance in batch:
                label = instance['Instance']['label']
                instruction = instance['Instance'][f'instruction_{lang}']

                # add bos and eos
                task_input = instruction
                label = label
                
                tokenized_input = self.tokenizer(task_input, add_special_tokens=False)["input_ids"]
                if len(tokenized_input)>self.max_source_length:
                    tokenized_input=tokenized_input[:self.max_source_length]

                tokenized_label = self.tokenizer(label, add_special_tokens=False)["input_ids"]
                if len(tokenized_label)>self.max_target_length:
                    tokenized_label=tokenized_label[:self.max_target_length]

                # (input) for inference, (input + label) for training
                if instance['subset'] in ['test']:
                    input_ids.append(tokenized_input)
                    labels.append([self.label_pad_token_id]*len(tokenized_input))
                else:
                    input_ids.append(tokenized_input)
                    labels.append([self.label_pad_token_id]*len(tokenized_input)+tokenized_label)
        
            inputs_length = [len(i) for i in input_ids]

            max_length = max(inputs_length)
            for i,l in enumerate(inputs_length):
                input_ids[i]=[self.tokenizer.pad_token_id]*(max_length-l) + input_ids[i]
                labels[i]=[self.label_pad_token_id]*(max_length-l) + labels[i]
                attention_mask.append([0]*(max_length-l) + [1]*l)

            model_inputs.update(
                {f'input_ids_{lang}': torch.tensor(input_ids),
                f'attention_mask_{lang}': torch.tensor(attention_mask),
                'labels': None}
                )

        return model_inputs

    # def get_instruction(self, instance):
    #     instruction = instance['Instance']["instruction"]
    #     content = instance['Instance']['sentence']

    #     # TODO, support few shot
    #     # add few shot samples
    #     samples = ''
    #     if len(instance['Samples']) > 0:
    #         raise Exception('Few shot is coming soon...')
    #     if samples:
    #         content = samples + content
    #     # TODO, fix bug

    #     return instruction


    def decoder_call(self, batch, return_tensors):
        self.tokenizer.padding_side = 'left'
        model_inputs = {}
        
        for lang in self.lang_list:
            input_ids, attention_mask, labels = [], [], []
            for instance in batch:
                label = instance['Instance']['label']
                instruction = instance['Instance'][f'instruction_{lang}']

                # add bos and eos
                task_input = instruction
                label = label
                
                tokenized_input = self.tokenizer(task_input, add_special_tokens=False)["input_ids"]
                if len(tokenized_input)>self.max_source_length:
                    tokenized_input=tokenized_input[:self.max_source_length]

                tokenized_label = self.tokenizer(label, add_special_tokens=False)["input_ids"]
                if len(tokenized_label)>self.max_target_length:
                    tokenized_label=tokenized_label[:self.max_target_length]

                # (input) for inference, (input + label) for training
                if instance['subset'] in ['test']:
                    input_ids.append(tokenized_input)
                    labels.append([self.label_pad_token_id]*len(tokenized_input))
                else:
                    input_ids.append(tokenized_input)
                    labels.append([self.label_pad_token_id]*len(tokenized_input)+tokenized_label)
        
            inputs_length = [len(i) for i in input_ids]

            max_length = max(inputs_length)
            for i,l in enumerate(inputs_length):
                input_ids[i]=[self.tokenizer.pad_token_id]*(max_length-l) + input_ids[i]
                labels[i]=[self.label_pad_token_id]*(max_length-l) + labels[i]
                attention_mask.append([0]*(max_length-l) + [1]*l)

            model_inputs.update(
                {f'input_ids_{lang}': torch.tensor(input_ids),
                f'attention_mask_{lang}': torch.tensor(attention_mask),
                'labels': None}
                )
        return model_inputs
