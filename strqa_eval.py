# Ref: https://github.com/kojima-takeshi188/zero_shot_cot

import re
import os
import json
import random
import torch
import numpy as np
import transformers
from tqdm import tqdm, trange
import argparse
from collections import defaultdict, Counter
import glob
import sys

import ssl
import urllib.request
import zipfile

from dola import DoLa

transformers.logging.set_verbosity(40)

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

N_SHOT = 6
COT_FLAG = True
DEBUG = False
ANSWER_TRIGGER = "So the answer is"
SHORT_ANSWER_TRIGGER = "answer is" # for long answer

def load_jsonl(file_path, is_gzip=False):
    # Format of each line in StrategyQA:
    # {"qid": ..., "term": ..., "description": ..., "question": ..., "answer": ..., "facts": [...], "decomposition": [...]}
    
    list_data_dict = []
    open_func = open if not is_gzip else gzip.open
    with open_func(file_path, 'r') as f:
        items = json.load(f)
        for item in items:
            new_item = dict(
                qid=item.get('qid', None),
                # term=item.get('term', None),
                # description=item.get('description', None),
                question=item.get('question', None),
                answer=item.get('answer', None),
                # facts=item.get('facts', []),
                # decomposition=item.get('decomposition', [])
            )
            list_data_dict.append(new_item)
    return list_data_dict

def download_url(url: str, folder='folder'):
    """
    Downloads the content of an url to a folder. Modified from \
    https://github.com/pyg-team/pytorch_geometric/tree/master/torch_geometric

    Args:
        url (string): The url of target file.
        folder (string): The target folder.

    Returns:
        string: File path of downloaded files.
    """

    file = url.rpartition('/')[2]
    file = file if file[0] == '?' else file.split('?')[0]
    path = os.path.join(folder, file)
    if os.path.exists(path):
        print(f'File {file} exists, use existing file.')
        return path

    print(f'Downloading {url}')
    os.makedirs(folder, exist_ok=True)
    ctx = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=ctx)
    with open(path, 'wb') as f:
        f.write(data.read())

    return path

def extract_answer_from_output(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_answer, answer):
    gt_answer = answer
    assert gt_answer != INVALID_ANS
    return model_answer == gt_answer

def create_demo_text(n_shot=6, cot_flag=True, shuffle=False):
    question, chain, answer = [], [], []
    question.append("Do hamsters provide food for any animals?")
    chain.append("Hamsters are prey animals. Prey are food for predators. Thus, hamsters provide food for some animals.")
    answer.append("yes")

    question.append("Could Brooke Shields succeed at University of Pennsylvania?")
    chain.append("Brooke Shields went to Princeton University. Princeton University is about as academically rigorous as the University of Pennsylvania. Thus, Brooke Shields could also succeed at the University of Pennsylvania.")
    answer.append("yes")

    question.append("Yes or no: Hydrogen's atomic number squared exceeds number of Spice Girls?")
    chain.append("Hydrogen has an atomic number of 1. 1 squared is 1. There are 5 Spice Girls. Thus, Hydrogen's atomic number squared is less than 5.")
    answer.append("no")

    question.append("Yes or no: Is it common to see frost during some college commencements?")
    chain.append("College commencement ceremonies can happen in December, May, and June. December is in the winter, so there can be frost. Thus, there could be frost at some commencements.")
    answer.append("yes")

    question.append("Yes or no: Could a llama birth twice during War in Vietnam (1945-46)?")
    chain.append("The War in Vietnam was 6 months. The gestation period for a llama is 11 months, which is more than 6 months. Thus, a llama could not give birth twice during the War in Vietnam.")
    answer.append("no")

    question.append("Yes or no: Would a pear sink in water?")
    chain.append("The density of a pear is about 0.6 g/cm^3, which is less than water. Objects less dense than water float. Thus, a pear would float.")
    answer.append("no")

    # randomize order of the examples ...
    index_list = list(range(len(question)))
    if shuffle:
        random.shuffle(index_list)

    # Concatenate demonstration examples ...
    demo_text = ""
    for i in index_list[:n_shot]:
        if cot_flag:
            demo_text += "Q: " + question[i] + "\nA: " + chain[i] + " " + \
                         ANSWER_TRIGGER + " " + answer[i] + ".\n\n"
        else:
            demo_text += "Question: " + question[i] + "\nAnswer: " + \
                         ANSWER_TRIGGER + " " + answer[i] + ".\n\n"
    return demo_text


def build_prompt(input_text, n_shot, cot_flag, shuffle):
    demo = create_demo_text(n_shot, cot_flag, shuffle)
    input_text_prompt = demo + "Q: " + input_text + "\n" + "A:"
    return input_text_prompt

def clean_answer(model_pred, random_guess=False):
    model_pred = model_pred.lower()

    if "Thus, yes." in model_pred:
        preds = "yes"
    elif SHORT_ANSWER_TRIGGER.lower() in model_pred:
        preds = model_pred.split(SHORT_ANSWER_TRIGGER.lower())[1].split(".")[0].strip()
    else:
        print("Warning: answer trigger not found in model prediction:", model_pred, "; returning yes/no based on exact match of `no`.", flush=True)
        if random_guess:
            preds = "no" if "no" in model_pred else "yes"
        else:
            return None
    if preds not in ["yes", "no"]:
        print("Warning: model prediction is not yes/no:", preds, "; returning no", flush=True)
        if random_guess:
            preds = "no"
        else:
            return None

    return (preds == "yes")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="huggyllama/llama-7b")
    parser.add_argument("--num-gpus", type=str, default="1")
    parser.add_argument("--max_gpu_memory", type=int, default=27)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--data-path", type=str, default="./strqa")
    parser.add_argument("--output-path", type=str, default="./strqa_result")
    # parallel mode (split the dataset into multiple parts, inference by separate processes)
    parser.add_argument("--early-exit-layers", type=str, default="-1")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--total-shard", type=int, default=8)
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--relative_top", type=float, default=0.1)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--do_shuffle", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retry", type=int, default=1)
    
    args = parser.parse_args()
    model_name = args.model_name
    num_gpus = args.num_gpus
    device = args.device

    # set seed
    set_seed(args.seed)

    # load your finetuned model (saved as xxx.ckpt)
    #    in yaml file federate.save_to

    # Get test file
    '''
    The StrategyQA dataset includes the followings files:
        strategyqa_train.json: The training set of StrategyQA, which includes 2,290 examples.
        strategyqa_train_paragraphs.json: Paragraphs from our corpus that were matched as evidence for examples in the training set.
        strategyqa_train_filtered.json: 2,821 additional questions, excluded from the official training set, that were filtered by our solvers during data collection (see more details in the paper).
        strategyqa_test.json: The test set of StrategyQA, which includes 490 examples.
    Here we only need the test set.
    '''
    fp = os.path.join(args.data_path, 'strategyqa_train.json')
    if not os.path.exists(fp):
        download_url(
            'https://storage.googleapis.com/ai2i/strategyqa/data/strategyqa_dataset.zip', args.data_path)

        # Once the file is downloaded, unzip it
        with zipfile.ZipFile(os.path.join(args.data_path, 'strategyqa_dataset.zip'), 'r') as zip_ref:
            zip_ref.extractall(args.data_path)

    list_data_dict = load_jsonl(fp)
    
    if args.parallel:
        chunk_size = len(list_data_dict) // args.total_shard
        list_data_dict = list_data_dict[args.shard_id * chunk_size: (args.shard_id + 1) * chunk_size]
    
    if args.debug:
        list_data_dict = list_data_dict[:10]
    
    llm = DoLa(model_name, device, num_gpus, args.max_gpu_memory)
    stop_word_list = ["Q:", "\n\n##"]
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
        mode = "dola-static"
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
    result_dict = {'is_correct': [], 'model_answer': [], 'model_completion': [], 'full_input_text': []}
    retry_times = args.retry
    for sample in tqdm(list_data_dict):
        model_answer = None
        for i in range(retry_times):
            input_text = build_prompt(sample['question'], N_SHOT, COT_FLAG, args.do_shuffle)
            generate_kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=args.do_sample, top_p=args.top_p, top_k=args.top_k, temperature=args.temperature, repetition_penalty=args.repetition_penalty, mode=mode, mature_layer=mature_layer, premature_layer=premature_layer, candidate_premature_layers=candidate_premature_layers, relative_top=args.relative_top)
            model_completion, c_dist = llm.generate(input_text, **generate_kwargs)
            for stop_word in stop_word_list:
                length_to_remove = len(stop_word)
                if model_completion[-length_to_remove:] == stop_word:
                    model_completion = model_completion[:-length_to_remove]
            model_completion = model_completion.strip()
            if mode == "dola":
                for k, v in c_dist.items():
                    premature_layer_dist[k] += v
            model_answer = clean_answer(model_completion, random_guess = (i == retry_times - 1))
            if model_answer is not None:
                break
        is_cor = is_correct(model_answer, sample['answer'])
        answers.append(is_cor)
        result_dict['is_correct'].append(is_cor)
        result_dict['model_answer'].append(model_answer)
        result_dict['model_completion'].append(model_completion)
        result_dict['full_input_text'].append(input_text)
        if DEBUG:
            print(f'Full input_text:\n{input_text}\n\n')
        print(f'Question: {sample["question"]}\n\n'
            f'Answers: {sample["answer"]}\n\n'
            f'Model Answers: {model_answer}\n\n'
            f'Model Completion: {model_completion}\n\n'
            f'Is correct: {is_cor}\n\n')

        print(f'Num of total question: {len(answers)}, '
            f'correct num: {sum(answers)}, '
            f'correct rate: {float(sum(answers))/len(answers)}.')

    if mode == "dola" and args.debug:
        total_tokens = sum(premature_layer_dist.values())
        if total_tokens > 0:
            for l in candidate_premature_layers:
                print('Premature layer {0} was used {1} times, {2}%'.format(l, premature_layer_dist[l], round(premature_layer_dist[l] / total_tokens * 100, 2)))
    # save results to a json file
    model_tag = model_name.split('/')[-1] if model_name[-1] != '/' else model_name.split('/')[-2]
    output_file = args.output_path if args.shard_id is None else (args.output_path+"_"+str(args.shard_id)+".json")
    with open(output_file, 'w') as f:
        json.dump(result_dict, f)
    print(f"{float(sum(answers))/len(answers)}")