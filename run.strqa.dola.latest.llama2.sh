python strqa_eval_migrate.py --early-exit-layers 0,2,4,6,8,10,12,14,32 --model-name /data/sls/d/llm/llama2/Llama-2-7b-hf --output-path latest.strqa.dola.latest.llama2.jsonl > log.strqa.dola.latest.llama2 2>&1
python strqa_eval_migrate.py --early-exit-layers -1 --model-name /data/sls/d/llm/llama2/Llama-2-7b-hf --output-path latest.strqa.van.latest.llama2.jsonl > log.strqa.van.latest.llama2 2>&1
