import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from conversation import get_default_conv_template
from dola import DoLa


def run_eval(llm, model_id, question_file, answer_file, generate_kwargs):
    ques_jsons = []
    with open(os.path.expanduser(question_file), "r") as ques_file:
        for line in ques_file:
            ques_jsons.append(line)

    ans_jsons = get_model_answers(llm, model_id, ques_jsons, generate_kwargs)

    with open(os.path.expanduser(answer_file), "w") as ans_file:
        for line in ans_jsons:
            ans_file.write(json.dumps(line) + "\n")


@torch.inference_mode()
def get_model_answers(llm, model_id, question_jsons, generate_kwargs):

    ans_jsons = []
    for i, line in enumerate(tqdm(question_jsons)):
        ques_json = json.loads(line)
        idx = ques_json["question_id"]
        qs = ques_json["text"]
        conv = get_default_conv_template(model_id).copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        outputs, c_dist = llm.generate(prompt, **generate_kwargs)

        ans_id = shortuuid.uuid()
        ans_jsons.append({"question_id": idx,
                          "text": outputs,
                          "answer_id": ans_id,
                          "model_id": model_id,
                          "metadata": {}})
    return ans_jsons


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answer-file", type=str, default="answer.jsonl")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--max_gpu_memory", type=int, default=27)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--early-exit-layers", type=str, default="-1")
    parser.add_argument("--relative_top", type=float, default=0.1)
    parser.add_argument("--do_sample", action="store_true")
    args = parser.parse_args()

    
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

    model_name = args.model_name
    num_gpus = args.num_gpus
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    llm = DoLa(model_name, device, num_gpus, args.max_gpu_memory)
    llm.set_stop_words(["### Human:"])
    generate_kwargs = dict(do_sample=True, max_new_tokens=1024, temperature=0.7, repetition_penalty=args.repetition_penalty, mode=mode, mature_layer=mature_layer, premature_layer=premature_layer, candidate_premature_layers=candidate_premature_layers, remove_stop_words=True, relative_top=args.relative_top)

    run_eval(llm, args.model_id, args.question_file, args.answer_file, generate_kwargs)
