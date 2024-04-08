# Ref: https://github.com/kojima-takeshi188/zero_shot_cot
# Ref: https://github.com/sylinrl/TruthfulQA/blob/main/truthfulqa/metrics.py
# Ref: https://github.com/sylinrl/TruthfulQA/blob/main/truthfulqa/utilities.py

import re
import os
import json
import random
import transformers
from tqdm import tqdm
import argparse
import pandas as pd

import ssl
import urllib.request
import zipfile

from dola_t5 import DoLa

transformers.logging.set_verbosity(40)

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

N_SHOT = 3
COT_FLAG = True
DEBUG = False
ANSWER_TRIGGER = "So the answer is"


def load_jsonl(file_path):
    with open(file_path) as f:
        list_prompts = [json.loads(line)['prompt'] for line in f]
    return list_prompts

def create_demo_text():
    question, answer = [], []

    question.append("Write a sentence describing the flavor of coffee. Make sure the word 'roasted' appears at least two times in the sentence, and include a bolded word. Like: *this is bolded text*.\"")
    answer.append("The bold, *roasted* flavor of coffee envelopes the palate, infusing each sip with rich, *roasted* notes reminiscent of toasted caramel and dark chocolate.")  # Based on answer_index -3

    question.append("List the months of the year using all capital letters.")
    answer.append("JANUARY, FEBRUARY, MARCH, APRIL, MAY, JUNE, JULY, AUGUST, SEPTEMBER, NOVEMBER, DECEMBER.") 

    demo_text = 'Take note of the instructions and responses in the following examples:' + '\n\n'
    for i in range(len(question)):
        demo_text += f'Example {i}: ' + "\nInstruction" + question[i] + "\nResponse" + answer[i] + "\n\n"
    return demo_text


def build_prompt(input_text):
    demo = create_demo_text()
    input_text_prompt = demo + "Now your task is: " + input_text
    return input_text_prompt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="huggyllama/llama-7b")
    parser.add_argument("--num-gpus", type=str, default="1")
    parser.add_argument("--max_gpu_memory", type=int, default=27)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--data-path", type=str, default="./tfqa")
    parser.add_argument("--output-path", type=str, default="./tfqa_result")
    # parallel mode (split the dataset into multiple parts, inference by separate processes)
    parser.add_argument("--early-exit-layers", type=str, default="-1")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--total-shard", type=int, default=8)
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument("--do-rating", action="store_true")
    parser.add_argument("--gpt3-config", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--relative_top", type=float, default=0.1)
    args = parser.parse_args()
    model_name = args.model_name
    num_gpus = args.num_gpus
    device = args.device
    data_path = args.data_path

    # Get test file
    fp = data_path + 'ifeval-input-data.jsonl'

    list_data_dict = load_jsonl(fp)

    if args.debug:
        list_data_dict = list_data_dict[:10]
    
    if args.parallel:
        chunk_size = len(list_data_dict) // args.total_shard
        list_data_dict = list_data_dict[args.shard_id * chunk_size: (args.shard_id + 1) * chunk_size]
    
    llm = DoLa(model_name, device, num_gpus, args.max_gpu_memory)

    # stop_word_list = ["Q:"]
    # llm.set_stop_words(stop_word_list)
    early_exit_layers = [int(x) for x in args.early_exit_layers.split(',')]
    if len(early_exit_layers) == 1:
        print("MODE: naive decoding from the last layer", flush=True)
        mode = "baseline"
        mature_layer = None
        premature_layer = None
        candidate_premature_layers = None
        if args.repetition_penalty is None:
            args.repetition_penalty = 1.2
    elif len(early_exit_layers) == 2:
        print(f"MODE: DoLa-static decoding with mature layer: {early_exit_layers[1]} and premature layer: {early_exit_layers[0]}")
        mode = "early_exit_contrastive"
        mature_layer = early_exit_layers[1]
        premature_layer = early_exit_layers[0]
        candidate_premature_layers = None
        if args.repetition_penalty is None:
            args.repetition_penalty = 1.2
    else:
        print(f"MODE: DoLa decoding with mature layer: {early_exit_layers[-1]} and premature layers: {early_exit_layers[:-1]}")
        mode = "dola"
        mature_layer = early_exit_layers[-1]
        premature_layer = None
        candidate_premature_layers = early_exit_layers[:-1]
        premature_layer_dist = {l:0 for l in candidate_premature_layers}
        if args.repetition_penalty is None:
            args.repetition_penalty = 1.2

    results = []
    for i, prompt in enumerate(tqdm(list_data_dict)):
        result_dict = {}

        # input_text = build_prompt(prompt)
        input_text = prompt
        generate_kwargs = dict(max_new_tokens=args.max_new_tokens, top_p=args.top_p, top_k=args.top_k, temperature=args.temperature, repetition_penalty=args.repetition_penalty, mode=mode, mature_layer=mature_layer, premature_layer=premature_layer, candidate_premature_layers=candidate_premature_layers)
        model_completion, c_dist = llm.generate(input_text, **generate_kwargs)
        
        # for stop_word in stop_word_list:
        #     length_to_remove = len(stop_word)
        #     if model_completion[-length_to_remove:] == stop_word:
        #         model_completion = model_completion[:-length_to_remove]
        model_completion = model_completion.strip()

        if mode == "dola":
            for k, v in c_dist.items():
                premature_layer_dist[k] += v

        result_dict['prompt'] = prompt
        result_dict['response'] = model_completion
        results.append(result_dict)
        
        if DEBUG:
            print(f'Full input_text:\n{input_text}\n\n')
        
        print(f'Question: {prompt}\n\n'
            f'Model Completion: {model_completion}\n\n')

        
    if mode == "dola" and args.debug:
        total_tokens = sum(premature_layer_dist.values())
        if total_tokens > 0:
            for l in candidate_premature_layers:
                print('Premature layer {0} was used {1} times, {2}%'.format(l, premature_layer_dist[l], round(premature_layer_dist[l] / total_tokens * 100, 2)))
    # save results to a json file
    model_tag = model_name.split('/')[-1] if model_name[-1] != '/' else model_name.split('/')[-2]
    output_file = args.output_path if args.shard_id is None else (args.output_path+"_"+str(args.shard_id)+".jsonl")

    # Write out in jsonl format
    with open(output_file, 'w') as f:
        for result in results:
            result_json_str = json.dumps(result)
            f.write(result_json_str + '\n')
    

    
        
