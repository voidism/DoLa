import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from conversation import get_default_conv_template
from open_ended_contrastive_early_exit import OpenEndedContrastiveEarlyExit


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
        # inputs = tokenizer([prompt])
        # output_ids = model.generate(
        #     torch.as_tensor(inputs.input_ids).cuda(),
        #     do_sample=True,
        #     temperature=0.7,
        #     max_new_tokens=1024)
        # outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        # skip_echo_len = len(prompt.replace("</s>", " ")) + 1

        # outputs = outputs[skip_echo_len:].strip()
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
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--early-exit-layers", type=str, default="-1")
    parser.add_argument("--divergence-type", type=str, default="js")
    parser.add_argument("--skip-layer0", action="store_true")
    parser.add_argument("--relative_top", type=float, default=0.1)
    parser.add_argument("--do_sample", action="store_true")
    args = parser.parse_args()

    
    early_exit_layers = [int(x) for x in args.early_exit_layers.split(',')]
    if early_exit_layers == [-1]:
        print("MODE: naive decoding from the last layer", flush=True)
        mode = "vanilla"
        final_layer = None
        base_layer = None
        dynamic_exit_layers = None
    elif len(early_exit_layers) == 2:
        print(f"MODE: early exit contrastive with final layer: {early_exit_layers[1]} and base layer: {early_exit_layers[0]}")
        mode = "early_exit_contrastive"
        final_layer = early_exit_layers[1]
        base_layer = early_exit_layers[0]
        dynamic_exit_layers = None
    else:
        print(f"MODE: dynamic early exit contrastive with final layer: {early_exit_layers[-1]} and base layers: {early_exit_layers[:-1]}")
        mode = "dynamic_early_exit_contrastive"
        final_layer = early_exit_layers[-1]
        base_layer = None
        dynamic_exit_layers = early_exit_layers[:-1]
        critical_layer_dist = {l:0 for l in dynamic_exit_layers}

    model_name = args.model_name
    num_gpus = args.num_gpus
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    llm = OpenEndedContrastiveEarlyExit(model_name, device, num_gpus)
    llm.set_stop_words(["### Human:"])
    generate_kwargs = dict(do_sample=True, max_new_tokens=1024, temperature=0.7, repetition_penalty=args.repetition_penalty, mode=mode, final_layer=final_layer, base_layer=base_layer, base_layers=dynamic_exit_layers, divergence_type=args.divergence_type, remove_stop_words=True, skip_layer0=args.skip_layer0, relative_top=args.relative_top)

    run_eval(llm, args.model_id, args.question_file, args.answer_file, generate_kwargs)
