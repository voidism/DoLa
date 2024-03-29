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

from dola_memotrap_dataset import DoLa

transformers.logging.set_verbosity(40)

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

N_SHOT = 7
COT_FLAG = True
DEBUG = False
ANSWER_TRIGGER = "So the answer is"


def parse_classes(classes_str):
    # Strip the outermost brackets and quotes
    classes_str = classes_str.strip('"[]')
    # Split the string into list items based on ', ' delimiter
    classes_list = classes_str.split("', '")
    # Strip extra whitespace and single quotes from each item
    classes_list = [item.strip().strip("'") for item in classes_list]
    return classes_list

def load_csv(file_path):

    list_data = []
    with open(file_path, 'r') as f:
        # Skip header
        next(f)
        for line in f:
            # Splitting by the first and last comma to isolate the prompt, classes, and answer_index
            first_comma = line.find(',')
            last_comma = line.rfind(',')
            prompt = line[:first_comma].strip('"')
            classes_str = line[first_comma + 1:last_comma].strip('"')
            answer_index = int(line[last_comma + 1:])

            # Use parse_classes function to convert classes string to list
            classes = parse_classes(classes_str)

            data = {'prompt': prompt, 'classes': classes, 'answer_index': answer_index}
            list_data.append(data)

    return list_data



def extract_and_compare_answer(question, model_completion):
    # Extract the ending of the model's completion
    # Assuming the prompt always ends with the start of the model's completion
    prompt_ending = question['prompt'].split()[-1]
    model_answer_ending = model_completion.split(prompt_ending)[-1].strip()

    # Get the correct answer based on the answer_index
    correct_answer = question['classes'][question['answer_index']].strip()

    # Compare the model's answer ending with the correct answer
    is_correct = model_answer_ending == correct_answer

    return is_correct, model_answer_ending, correct_answer


def create_demo_text():
    question, answer = [], []
    
    question.append("Write a quote that ends in the word \"men\": Young men think old men fools, and old men know young men to be")
    answer.append("Young men think old men fools, and old men know young men to be men.")  # Based on answer_index 0

    question.append("Write a quote that ends in the word \"man\": Young saint, old")
    answer.append("Young saint, old man.")  # Based on answer_index 1

    question.append("Write a quote that ends in the word \"change\": Youth and age will never")
    answer.append("Youth and age will never change.")  # Based on answer_index 1


    demo_text = 'Consider the ending word of each quote and complete it, pay attention to the instructions you are being asked to follow.' + '\n\n'
    for i in range(len(question)):
        demo_text += "Q: " + question[i] + "\nA: " + answer[i] + "\n\n"
    return demo_text


def build_prompt(input_text):
    demo = create_demo_text()
    input_text_prompt = demo + "Q: " + input_text + "\n" + "A:"
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
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--relative_top", type=float, default=0.1)
    args = parser.parse_args()
    model_name = args.model_name
    num_gpus = args.num_gpus
    device = args.device

    # Get test file
 
    fp = '1-proverb-ending.csv'

    list_data_dict = load_csv(fp)

    if args.debug:
        list_data_dict = list_data_dict[:10]
    
    if args.parallel:
        chunk_size = len(list_data_dict) // args.total_shard
        list_data_dict = list_data_dict[args.shard_id * chunk_size: (args.shard_id + 1) * chunk_size]
    
    llm = DoLa(model_name, device, num_gpus, args.max_gpu_memory)
    stop_word_list = ["Q:"]
    llm.set_stop_words(stop_word_list)
    early_exit_layers = [int(x) for x in args.early_exit_layers.split(',')]
    if len(early_exit_layers) == 1:
        print("MODE: naive decoding from the last layer", flush=True)
        mode = "baseline"
        mature_layer = None
        premature_layer = None
        candidate_premature_layers = None
        if args.repetition_penalty is None:
            args.repetition_penalty = 1.0
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

    answers = []
    result_dict = {'question': [], 'model_completion': [], 'model_answer_ending': [], 'correct_answer': [], 'correctness': []}
    for i, sample in enumerate(tqdm(list_data_dict)):
        
        input_text = build_prompt(sample['prompt'])
        generate_kwargs = dict(max_new_tokens=args.max_new_tokens, top_p=args.top_p, top_k=args.top_k, temperature=args.temperature, repetition_penalty=args.repetition_penalty, mode=mode, mature_layer=mature_layer, premature_layer=premature_layer, candidate_premature_layers=candidate_premature_layers)
        model_completion, c_dist = llm.generate(input_text, **generate_kwargs)
        
        
        for stop_word in stop_word_list:
            length_to_remove = len(stop_word)
            if model_completion[-length_to_remove:] == stop_word:
                model_completion = model_completion[:-length_to_remove]
        model_completion = model_completion.strip()
        
        is_correct, model_answer_ending, correct_answer = extract_and_compare_answer(question=sample, model_completion=model_completion)
        
        if mode == "dola":
            for k, v in c_dist.items():
                premature_layer_dist[k] += v
        model_answer = model_completion
        result_dict['model_completion'].append(model_completion)
        result_dict['question'].append(sample)
        result_dict['model_answer_ending'].append(model_answer_ending)
        result_dict['correct_answer'].append(correct_answer)
        result_dict['correctness'].append(is_correct)
        
        if DEBUG:
            print(f'Full input_text:\n{input_text}\n\n')
        
        print(f'Question: {sample}\n\n'
            f'Model Completion: {model_completion}\n\n')

        
    if mode == "dola" and args.debug:
        total_tokens = sum(premature_layer_dist.values())
        if total_tokens > 0:
            for l in candidate_premature_layers:
                print('Premature layer {0} was used {1} times, {2}%'.format(l, premature_layer_dist[l], round(premature_layer_dist[l] / total_tokens * 100, 2)))
    # save results to a json file
    model_tag = model_name.split('/')[-1] if model_name[-1] != '/' else model_name.split('/')[-2]
    output_file = args.output_path if args.shard_id is None else (args.output_path+"_"+str(args.shard_id)+".jsonl")
    with open(output_file, 'w') as f:
        json.dump(result_dict, f)
    

    
        