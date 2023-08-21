DoLa: Decoding by Contrasting Layers
===

Code for the paper "Decoding in the Depths: Contrasting Layerwise Knowledge Improves Factuality in Large Language Models"

## Setup

```
pip install -e transformers-4.28.1
pip install datasets
pip install accelerate
pip install openai # -> only for truthfulqa and gpt4_eval
```

## Run


### Arguments

| Argument        | Example           | Description   |
| --------------- | ----------------- | ------------- |
| `--model-name`  | `huggyllama/llama-7b` | Specifies the model you want to use |
| `--data-path`   | `/path/to/dataset` | Path to the dataset file |
| `--output-path` | `output-path.jsonl` | Where to store the output results |
| `--num-gpus`    | `1` | Number of GPUs to use, `1/2/4/8` for `7B/13B/30B/65B` model sizes respectively.  |

### Understanding `--early-exit-layers`

The `--early-exit-layers` argument takes a string containing a sequence of numbers separated by commas, with no spaces in between. For example, 


| Number of Layers Specified  | Example (str)     | Description                                                                                     |
| ---------------------------| ------------- | ----------------------------------------------------------------------------------------------- |
| 1                          | `-1`      | **Naive decoding** from the last layer.       |
| 2                          | `16,32`   | **DoLa-static decoding** with the second layer (i.e. `32`) as the `mature_layer` and first layer (i.e. `16`) as `premature_layer`. |
| >2                         | `0,2,4,6,8,10,12,14,32`    | **DoLa decoding** with the last specified layer (i.e. `32`) as the `mature_layer` and all the preceding layers (i.e. `0,2,4,6,8,10,12,14`) as `candidate_premature_layers`. |

### FACTOR
Please contact the author of [Generating Benchmarks for Factuality Evaluation of Language Models](https://arxiv.org/abs/2307.06908) to get early released dataset, before the dataset is released at https://github.com/AI21Labs/factor.

#### Baseline
```bash
python factor_eval.py --model-name huggyllama/llama-7b --data-path /path/to/wiki_factor.csv --output-path output-path.jsonl --num-gpus 1
python factor_eval.py --model-name huggyllama/llama-13b --data-path /path/to/wiki_factor.csv --output-path output-path.jsonl --num-gpus 2
python factor_eval.py --model-name huggyllama/llama-30b --data-path /path/to/wiki_factor.csv --output-path output-path.jsonl --num-gpus 4
python factor_eval.py --model-name huggyllama/llama-65b --data-path /path/to/wiki_factor.csv --output-path output-path.jsonl --num-gpus 8
```

#### DoLa
```bash
python factor_eval.py --model-name huggyllama/llama-7b --early-exit-layers 0,2,4,6,8,10,12,14,32 --data-path /path/to/wiki_factor.csv --output-path output-path.jsonl --num-gpus 1
python factor_eval.py --model-name huggyllama/llama-13b --early-exit-layers 0,2,4,6,8,10,12,14,16,18,40 --data-path /path/to/wiki_factor.csv --output-path output-path.jsonl --num-gpus 2
python factor_eval.py --model-name huggyllama/llama-30b --early-exit-layers 0,2,4,6,8,10,12,14,16,18,60 --data-path /path/to/wiki_factor.csv --output-path output-path.jsonl --num-gpus 4
python factor_eval.py --model-name huggyllama/llama-65b --early-exit-layers 0,2,4,6,8,10,12,14,16,18,80 --data-path /path/to/wiki_factor.csv --output-path output-path.jsonl --num-gpus 8
```

Change `wiki_factor.csv` to `news_factor.csv` for the news subset of FACTOR.

### TruthfulQA-MC

### TruthfulQA

### StrategyQA

### GSM8K

### GPT-4 Evaluation (Vicuna QA Benchmark)



## Reference Repositories
- FastChat: https://github.com/lm-sys/FastChat
- ContrastiveDecoding: https://github.com/XiangLi1999/ContrastiveDecoding
- TruthfulQA: https://github.com/sylinrl/TruthfulQA
- zero_shot_cot: https://github.com/kojima-takeshi188/zero_shot_cot
- FederatedScope: https://github.com/alibaba/FederatedScope

