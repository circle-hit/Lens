import torch
from transformers import GenerationConfig
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.trainer import *
import numpy as np
import torch.nn as nn
import pickle

from collator import SUPPORTED_DECODER_MODELS, check_model
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM

def nested_truncate(tensors, limit):
    "Truncate `tensors` at `limit` (even if it's a nested list/tuple of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_truncate(t, limit) for t in tensors)
    return tensors[:limit]

def skip_instructions(model, predictions_ids, tokenizer, ignore_idx=-100):
    predictions_ids = np.where(predictions_ids == ignore_idx, tokenizer.pad_token_id, predictions_ids)
    predictions = tokenizer.batch_decode(
        predictions_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    return predictions

def create_memory_replay_generators(task, task_list, replay_data_dict, split='train_mem'): # creating previous tasks memory buffers
    print('Creating generators for previous tasks ...')
    tasks_to_generators = {}
    curr_task_num = task_list.index(task)
    for idx in np.arange(curr_task_num):
        prev_task = task_list[idx]
        tasks_to_generators[prev_task] = iter(replay_data_dict[prev_task])
    return tasks_to_generators

class Trainer(Seq2SeqTrainer):
    
    def project_into_vocabluary(self, vector, E, tokenizer, top_k=20, bottom_k=-1):
        """
        Project a vector into the vocabulary space and return the top_k tokens.
        :param vector: D dimensional vector
        :param E: Language model embedding matrix (V, D)
        :param tokenizer: Model tokenizer
        :param top_k: How many top tokens to return
        :param bottom_k: How many bottom tokens to return. If -1, return top_k tokens
        :return:
        """
        vector = vector.to(torch.float32).to('cuda')
        E = E.to(torch.float32).to('cuda')
        vocab_ranking = torch.matmul(E, vector)     # (V,)
        sorted_token_ids = np.argsort(vocab_ranking.detach().cpu().numpy())[::-1]  # Descending order
        if bottom_k == -1:
            sorted_tokens = [tokenizer.decode(x).strip() for x in sorted_token_ids[:top_k]]
        else :
            sorted_tokens = [tokenizer.decode(x).strip() for x in sorted_token_ids[-bottom_k:][::-1]]  # Least score to most score
        return sorted_tokens

    def __init__(self, model, args, train_dataset, tokenizer=None, data_collator=None):
        super().__init__(model=model, args=args, train_dataset=train_dataset, tokenizer=tokenizer, data_collator=data_collator)
        
        if self.args.retain_weight > 0:
            print("Loading Reference Model ...")
            self.reference_model = AutoModelForCausalLM.from_pretrained(
                self.args.model_name_or_path,
                use_safetensors=True,
                torch_dtype="auto",
            )
            self.reference_model.to(self.args.device).to(torch.bfloat16)
        
        print(f"Loading Language Space from {args.space_tag}")
        with open(f'Probing/language_space/{args.model_id.split("-")[0]}/{args.space_tag}/lang_specific_space_last.pkl', 'rb') as f:
            self.lang_spec_space = pickle.load(f)
        
        with open(f'Probing/language_space/{args.model_id.split("-")[0]}/{args.space_tag}/lang_shared_space_last.pkl', 'rb') as f:
            self.lang_comm_space = pickle.load(f)
        
        self.lang_direction = []
        
        for lang in args.lang_list:
            with open(f'Probing/language_space/{args.model_id.split("-")[0]}/{args.space_tag}/{lang}_direction_last.pkl', 'rb') as f:
                lang_dir = torch.stack(pickle.load(f))
                lang_dir.requires_grad = False
                self.lang_direction.append(lang_dir)

    def projection(self, emb, lang_dir):
        
        lang_dir_norm = lang_dir / torch.linalg.norm(lang_dir, axis=1, keepdims=True).to(emb.dtype)
        proj = torch.matmul(emb, lang_dir_norm.T)
        
        return torch.matmul(proj, lang_dir_norm)

    def orthogonality_loss(self, u, v):
        # 计算每一对向量的内积
        inner_product = torch.sum(u * v, dim=1)  # (B,)

        # 计算正交性损失（内积的平方和）
        loss = torch.mean(inner_product ** 2)  # 标量

        return loss

    def compute_loss(self, model, inputs):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        
        outputs = []
        for lang in self.args.lang_list:
            outputs.append(model(**{'input_ids': inputs[f"input_ids_{lang}"], 'attention_mask': inputs[f"attention_mask_{lang}"]}, output_hidden_states=True))

        if self.args.retain_weight > 0:
            outputs_ref = []
            for lang in self.args.lang_list:
                with torch.no_grad():
                    outputs_ref.append(self.reference_model(**{'input_ids': inputs[f"input_ids_{lang}"], 'attention_mask': inputs[f"attention_mask_{lang}"]}, output_hidden_states=True))
            
            en_retain_loss = nn.MSELoss()(outputs[0].hidden_states[-1].mean(dim=1), outputs_ref[0].hidden_states[-1].mean(dim=1))
            print(f"{self.args.retain_weight} * en_retain_loss:", self.args.retain_weight * en_retain_loss.item())

            loss = self.args.retain_weight * en_retain_loss
        
        if self.args.pull_weight['en'] > 0:
            lang_spec_proj_list = []
            for lang, outputs_lang, outputs_lang_ref, lang_direction in zip(self.args.lang_list[1:], outputs[1:], outputs_ref[1:], self.lang_direction[1:]):
                lang_spec_proj = self.projection(outputs_lang.hidden_states[-1][:, -1, :], self.lang_spec_space[-1].to(outputs_lang.hidden_states[-1].dtype).to(outputs_lang.hidden_states[-1].device))
                ori_lang_spec_proj = self.projection(outputs_lang_ref.hidden_states[-1][:, -1, :], self.lang_spec_space[-1].to(outputs_lang.hidden_states[-1].dtype).to(outputs_lang.hidden_states[-1].device))
                pull_loss = nn.MSELoss()(lang_spec_proj - ori_lang_spec_proj, self.args.pull_weight[lang] * lang_direction[-1].repeat(lang_spec_proj.shape[0], 1).to(outputs_lang.hidden_states[-1].dtype).to(outputs_lang.hidden_states[-1].device))
                loss += pull_loss
                print(f"{self.args.pull_weight[lang]} * {lang}_pull_loss:", self.args.pull_weight[lang] * pull_loss.item()) 

                lang_spec_proj_list.append(lang_spec_proj)
        
        if self.args.push_weight > 0:
            # en_lang_spec_proj = self.projection(outputs[0].hidden_states[-1][:, -1, :], self.lang_spec_space[-1].to(outputs_lang.hidden_states[-1].dtype).to(outputs_lang.hidden_states[-1].device))
            for lang, outputs_lang, lang_spec_proj in zip(self.args.lang_list[1:], outputs[1:], lang_spec_proj_list):
                en_comm_proj = self.projection(outputs[0].hidden_states[-1][:, -1, :], self.lang_comm_space[-1].unsqueeze(dim=0).to(outputs[0].hidden_states[-1].dtype).to(outputs[0].hidden_states[-1].device))
                lang_comm_proj = self.projection(outputs_lang.hidden_states[-1][:, -1, :], self.lang_comm_space[-1].unsqueeze(dim=0).to(outputs_lang.hidden_states[-1].dtype).to(outputs_lang.hidden_states[-1].device))

                push_loss = nn.MSELoss()(en_comm_proj, lang_comm_proj)
                loss += self.args.push_weight * push_loss
                print(f"{self.args.push_weight} * {lang}_push_loss:", self.args.push_weight * push_loss.item())

        
        return loss

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
            
        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss, retain_graph=True)

        return loss.detach() / self.args.gradient_accumulation_steps

    