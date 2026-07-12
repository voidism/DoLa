from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import torch

tokenizer = AutoTokenizer.from_pretrained("/data/sls/d/llm/llama1/llama-7b")
model = AutoModelForCausalLM.from_pretrained("/data/sls/d/llm/llama1/llama-7b", torch_dtype=torch.float16)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
set_seed(42)

while True:
    text = input("Prompt: ")
    custom_layers = input("Layers: ")
    custom_layers = [int(x) for x in custom_layers.split(',')]
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    vanilla_output = model.generate(input_ids, do_sample=False, max_new_tokens=50)
    dola_low_output = model.generate(input_ids, do_sample=False, max_new_tokens=50, dola_layers='low')
    dola_high_output = model.generate(input_ids, do_sample=False, max_new_tokens=50, dola_layers='high')
    dola_custom_output = model.generate(input_ids, do_sample=False, max_new_tokens=50, dola_layers=custom_layers)

    # # skip the tokens in the input prompt
    vanilla_output = vanilla_output[0, input_ids.shape[-1]:].cpu().numpy()
    dola_low_output = dola_low_output[0, input_ids.shape[-1]:].cpu().numpy()
    dola_high_output = dola_high_output[0, input_ids.shape[-1]:].cpu().numpy()
    dola_custom_output = dola_custom_output[0, input_ids.shape[-1]:].cpu().numpy()

    vanilla_text = tokenizer.decode(vanilla_output, skip_special_tokens=True)
    dola_low_text = tokenizer.decode(dola_low_output, skip_special_tokens=True)
    dola_high_text = tokenizer.decode(dola_high_output, skip_special_tokens=True)
    dola_custom_text = tokenizer.decode(dola_custom_output, skip_special_tokens=True)
    print(f"Vanilla: {vanilla_text}", end='\n\n')
    print(f"Dola Low: {dola_low_text}", end='\n\n')
    print(f"Dola High: {dola_high_text}", end='\n\n')
    print(f"Dola Custom: {dola_custom_text}", end='\n\n')
