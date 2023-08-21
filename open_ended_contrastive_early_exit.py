import argparse
import time
import csv
import tqdm
import os
import json

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from transformers.generation.stopping_criteria import StoppingCriteriaList, LLamaQaStoppingCriteria

import argparse
import warnings
import pandas as pd
import numpy as np

class OpenEndedContrastiveEarlyExit:
    def __init__(self, model_name, device, num_gpus):
        self.model_name = model_name
        self.device = device
        self.num_gpus = num_gpus
        self.stopping_criteria = None

        self.model, self.tokenizer = self.load_model(model_name)

    def load_model(self, model_name):
        if self.device == "cuda":
            kwargs = {"torch_dtype": torch.float16, "offload_folder": f"{model_name}/offload"}
            if self.num_gpus == "auto":
                kwargs["device_map"] = "auto"
            else:
                self.num_gpus = int(self.num_gpus)
                if self.num_gpus != 1:
                    kwargs.update({
                        "device_map": "auto",
                        "max_memory": {i: "27GiB" for i in range(self.num_gpus)},
                    })
        elif self.device == "cpu":
            kwargs = {}
        else:
            raise ValueError(f"Invalid device: {self.device}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name if not 'vicuna' in model_name else 'huggyllama/llama-7b')
        model = AutoModelForCausalLM.from_pretrained(model_name,
            low_cpu_mem_usage=True, **kwargs)

        if self.device == "cuda" and self.num_gpus == 1:
            model.cuda()
        
        return model, tokenizer

    def set_stop_words(self, stop_words):
        self.stop_words = stop_words
        self.stopping_criteria = StoppingCriteriaList()
        list_stop_word_ids = []
        for stop_word in self.stop_words:
            stop_word_ids = self.tokenizer.encode('\n' + stop_word)[3:]
            list_stop_word_ids.append(stop_word_ids)
            print("Added stop word: ", stop_word, 'with the ids', stop_word_ids, flush=True)
        self.stopping_criteria.append(LLamaQaStoppingCriteria(list_stop_word_ids))

    def generate(self, input_text, max_new_tokens=256, top_p=0.95, top_k=0, temperature=0.8, final_layer=None, base_layer=None, base_layers=[], divergence_type='js', mode='vanilla', verbose=True, remove_stop_words=False, skip_layer0=False, relative_top=0.1, relative_top_with_norm=False, contrast_disagree_only=False, **kwargs):
        with torch.no_grad():

            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            max_len = input_ids.shape[-1] + max_new_tokens

            if mode == 'vanilla':
                outputs = self.model.generate(input_ids, max_length=max_len, num_return_sequences=1,
                                    output_scores=True, return_dict_in_generate=True, early_exit_contrastive_decoding=False,
                                    top_p=top_p, top_k=top_k, temperature=temperature, stopping_criteria=self.stopping_criteria, **kwargs)
            elif mode == 'early_exit_contrastive':
                assert final_layer is not None, "final_layer must be specified"
                assert base_layer is not None, "base_layer must be specified"
                outputs = self.model.generate(input_ids, max_length=max_len, num_return_sequences=1,
                                    output_scores=True, return_dict_in_generate=True, early_exit_contrastive_decoding=True,
                                    final_layer=final_layer, base_layer=base_layer,
                                    top_p=top_p, top_k=top_k, temperature=temperature, stopping_criteria=self.stopping_criteria, skip_layer0=skip_layer0, relative_top=relative_top, relative_top_with_norm=relative_top_with_norm, contrast_disagree_only=contrast_disagree_only, **kwargs)
            elif mode == 'dynamic_early_exit_contrastive':
                assert final_layer is not None, "final_layer must be specified"
                assert base_layers is not None, "base_layers must be specified"
                outputs = self.model.generate(input_ids, max_length=max_len, num_return_sequences=1,
                                        output_scores=True, return_dict_in_generate=True, early_exit_contrastive_decoding=True,
                                        top_p=top_p, top_k=top_k, temperature=temperature, stopping_criteria=self.stopping_criteria, skip_layer0=skip_layer0, relative_top=relative_top, relative_top_with_norm=relative_top_with_norm, contrast_disagree_only=contrast_disagree_only, 
                                        final_layer=final_layer, base_layer=None, dynamic_exit_layers=base_layers,
                                        divergence_type=divergence_type, **kwargs,)
                critical_layer_dist = outputs.critical_layer_dist
            sequences, scores = outputs.sequences, outputs.scores

            # skip the tokens in the input prompt
            gen_sequences = sequences[:, input_ids.shape[-1]:][0, :]
            gen_arr = gen_sequences.cpu().numpy()

            output_str = self.tokenizer.decode(gen_sequences, skip_special_tokens=True)

            if verbose:
                print('MODEL OUTPUT: \n{0}'.format(output_str))

            if remove_stop_words:
                for stop_word in self.stop_words:
                    length_to_remove = len(stop_word)
                    if output_str[-length_to_remove:] == stop_word:
                        output_str = output_str[:-length_to_remove]
                output_str = output_str.strip()

        if self.device:
            torch.cuda.empty_cache()

        return output_str, (critical_layer_dist if mode == 'dynamic_early_exit_contrastive' else None)

    def get_relative_top_filter(self, scores: torch.FloatTensor, relative_top: float = 0.1, min_tokens_to_keep: int = 1):
        scores_normalized = scores.log_softmax(dim=-1) 
        sorted_logits, sorted_indices = torch.sort(scores_normalized, descending=True)
        min_thresh = sorted_logits[..., min_tokens_to_keep-1] 
        probs_max = torch.max(scores_normalized, dim=-1).values
        probs_thresh = probs_max + np.log(relative_top)
        probs_thresh = torch.min(min_thresh, probs_thresh)
        probs_thresh = probs_thresh.unsqueeze(-1)
        return scores_normalized < probs_thresh

    def lm_score(self, input_text1, input_text2, pmi=False, max_new_tokens=256, top_p=0.95, top_k=0, temperature=0.8, final_layer=None, base_layer=None, base_layers=[], divergence_type='js', mode='vanilla', verbose=True, remove_stop_words=False, skip_layer0=False, relative_top=0.1, relative_top_with_norm=False, relative_top_value=-1000.0, contrast_disagree_only=False, extrapolate_coeff=None, **kwargs):
        with torch.no_grad():
            input_text = input_text1 + input_text2
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            prefix_ids = self.tokenizer(input_text1, return_tensors="pt").input_ids.to(self.device)
            continue_ids = input_ids[0, prefix_ids.shape[-1]:]
            if mode == 'vanilla':
                outputs = self.model(input_ids)[0].squeeze(0)
                outputs = outputs.log_softmax(-1)  # logits to log probs

                # skip tokens in the prompt -- we only care about the answer
                outputs = outputs[prefix_ids.shape[-1] - 1: -1, :]

                # get logprobs for each token in the answer
                log_probs = outputs[range(outputs.shape[0]), continue_ids].sum().item()

                # pmi
                if pmi:
                    outputs_y = self.model(input_ids[:, prefix_ids.shape[-1]-1:])[0].squeeze(0)[:-1]
                    outputs_y = outputs_y.log_softmax(-1)
                    log_probs_y = outputs_y[range(outputs_y.shape[0]), continue_ids].sum().item()
                    log_probs = log_probs - log_probs_y
                
            elif mode == 'early_exit_contrastive':
                dict_outputs, outputs = self.model(
                    input_ids=input_ids,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    early_exit_layers=[base_layer, final_layer],
                )

                assert base_layer is not None
                base_logits = dict_outputs[base_layer][0, prefix_ids.shape[-1] - 1: -1, :]
                final_logits = dict_outputs[final_layer][0, prefix_ids.shape[-1] - 1: -1, :]
                final_logits = final_logits.log_softmax(dim=-1)
                base_logits = base_logits.log_softmax(dim=-1)
                if extrapolate_coeff is None or extrapolate_coeff >= 1000.0:
                    diff_logits = final_logits - base_logits
                else:
                    diff_logits = base_logits + extrapolate_coeff * (final_logits - base_logits)
                if relative_top > 0.0:
                    relative_top_mask = self.get_relative_top_filter(final_logits, relative_top)
                    diff_logits = torch.where(relative_top_mask, relative_top_value, diff_logits)
                    
                log_probs = diff_logits[range(diff_logits.shape[0]), continue_ids].sum().item()

                # pmi
                if pmi:
                    dict_outputs_y, outputs_y = self.model(
                        input_ids=input_ids[:, prefix_ids.shape[-1]-1:],
                        return_dict=True,
                        output_attentions=False,
                        output_hidden_states=False,
                        early_exit_layers=[base_layer, final_layer],
                    )
                    base_logits_y = dict_outputs_y[base_layer][0, :, :]
                    final_logits_y = dict_outputs_y[final_layer][0, :, :]
                    final_logits_y = final_logits_y.log_softmax(dim=-1)
                    base_logits_y = base_logits_y.log_softmax(dim=-1)
                    diff_logits_y = final_logits_y - base_logits_y
                    if relative_top > 0.0:
                        relative_top_mask = self.get_relative_top_filter(final_logits_y, relative_top)
                        diff_logits_y = torch.where(relative_top_mask, relative_top_value, diff_logits_y)
                    # diff_logits_y = diff_logits_y.log_softmax(dim=-1)
                    log_probs_y = diff_logits_y[range(diff_logits_y.shape[0]), continue_ids].sum().item()
                    log_probs = log_probs - log_probs_y

            elif mode == 'early_exit_contrastive_exploit':
                dict_outputs, outputs = self.model(
                    input_ids=input_ids,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    early_exit_layers=base_layers + [final_layer],
                )

                return_dict = {}
                for base_layer in base_layers:
                    base_logits = dict_outputs[base_layer][0, prefix_ids.shape[-1] - 1: -1, :]
                    final_logits = dict_outputs[final_layer][0, prefix_ids.shape[-1] - 1: -1, :]
                    final_logits = final_logits.log_softmax(dim=-1)
                    base_logits = base_logits.log_softmax(dim=-1)
                    if extrapolate_coeff is None or extrapolate_coeff >= 1000.0:
                        diff_logits = final_logits - base_logits
                    else:
                        diff_logits = base_logits + extrapolate_coeff * (final_logits - base_logits)
                    if relative_top > 0.0:
                        relative_top_mask = self.get_relative_top_filter(final_logits, relative_top)
                        diff_logits = torch.where(relative_top_mask, relative_top_value, diff_logits)
                        
                    log_probs = diff_logits[range(diff_logits.shape[0]), continue_ids].sum().item()

                    # pmi
                    if pmi:
                        dict_outputs_y, outputs_y = self.model(
                            input_ids=input_ids[:, prefix_ids.shape[-1]-1:],
                            return_dict=True,
                            output_attentions=False,
                            output_hidden_states=False,
                            early_exit_layers=[base_layer, final_layer],
                        )
                        base_logits_y = dict_outputs_y[base_layer][0, :, :]
                        final_logits_y = dict_outputs_y[final_layer][0, :, :]
                        final_logits_y = final_logits_y.log_softmax(dim=-1)
                        base_logits_y = base_logits_y.log_softmax(dim=-1)
                        diff_logits_y = final_logits_y - base_logits_y
                        if relative_top > 0.0:
                            relative_top_mask = self.get_relative_top_filter(final_logits_y, relative_top)
                            diff_logits_y = torch.where(relative_top_mask, relative_top_value, diff_logits_y)
                        diff_logits_y = diff_logits_y.log_softmax(dim=-1)
                        log_probs_y = diff_logits_y[range(diff_logits_y.shape[0]), continue_ids].sum().item()
                        log_probs = log_probs - log_probs_y

                    return_dict[base_layer] = log_probs
                log_probs = return_dict

            elif mode == 'dynamic_early_exit_contrastive':
                critical_layer_dist = {l:0 for l in base_layers}
                picked_logits = []
                result_dict = {}
                critical_layers = []

                dict_outputs, outputs = self.model(
                    input_ids=input_ids,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    early_exit_layers=base_layers + [final_layer],
                )

                for seq_i in range(prefix_ids.shape[-1] - 1, input_ids.shape[-1] - 1):
                    # pick the less like layer to contrast with
                    kl_divs = torch.stack(
                        # reverse KL-divergence
                        [F.kl_div(F.log_softmax(dict_outputs[i][:, seq_i, :], dim=-1), F.softmax(dict_outputs[final_layer][:, seq_i, :], dim=-1), reduction='batchmean') for i in base_layers] if divergence_type == 'rev_kl' else (
                        # KL-divergence
                        [F.kl_div(F.log_softmax(dict_outputs[final_layer][:, seq_i, :], dim=-1), F.softmax(dict_outputs[i][:, seq_i, :], dim=-1), reduction='batchmean') for i in base_layers] if divergence_type == 'kl' else 
                        # JS-divergence
                        [0.5 * F.kl_div(F.log_softmax(dict_outputs[final_layer][:, seq_i, :], dim=-1), F.softmax(dict_outputs[i][:, seq_i, :], dim=-1), reduction='batchmean') + 0.5 * F.kl_div(F.log_softmax(dict_outputs[i][:, seq_i, :], dim=-1), F.softmax(dict_outputs[final_layer][:, seq_i, :], dim=-1), reduction='batchmean') for i in base_layers]
                        )
                    ).squeeze(-1)
                    critical_layer = base_layers[int(kl_divs.argmax().cpu().item())]
                    critical_layer_dist[critical_layer] += 1

                    critical_layers.append(critical_layer)

                base_logits = torch.zeros_like(dict_outputs[final_layer][0, prefix_ids.shape[-1] - 1:-1])
                for i, l in enumerate(critical_layers):
                   base_logits[i] = dict_outputs[l][0, prefix_ids.shape[-1] - 1 + i]
                final_logits = dict_outputs[final_layer][0, prefix_ids.shape[-1] - 1:-1]
                final_logits = final_logits.log_softmax(dim=-1)
                base_logits = base_logits.log_softmax(dim=-1)
                if extrapolate_coeff is None or extrapolate_coeff >= 1000.0:
                    diff_logits = final_logits - base_logits
                else:
                    diff_logits = base_logits + extrapolate_coeff * (final_logits - base_logits)

                if relative_top > 0.0:
                    relative_top_mask = self.get_relative_top_filter(final_logits, relative_top)
                    diff_logits = torch.where(relative_top_mask, relative_top_value, diff_logits)
                
                log_probs = diff_logits[range(diff_logits.shape[0]), continue_ids].sum().item()

        return log_probs, (critical_layer_dist if mode == 'dynamic_early_exit_contrastive' else None)