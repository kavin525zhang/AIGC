from tokenizer import JieBaTokenizer, CPTTokenizer
from transformers import T5ForConditionalGeneration, AutoModelForCausalLM, LlamaTokenizer
import torch
from models import CPTForConditionalGeneration


#model_path = '../models/t5-pegasus/'
model_path = '../models/cpt-large'
#model_path = "/data9/NFS/patent/model_hub/kavin_tmp/llama-7b-best/"

#checkpoints_dir = "./output_t5/fold=00-epoch=09-bleu=0.5939-rouge=0.7174-rouge=rouge-1=0.7786-rouge=rouge-2=0.6643-rouge=rouge-l=0.7400.ckpt"
checkpoints_dir = "./output_cpt/fold=00-epoch=01-bleu=0.3409-rouge=0.5768-rouge=rouge-1=0.6953-rouge=rouge-2=0.4864-rouge=rouge-l=0.6080.ckpt"
checkpoint = torch.load(checkpoints_dir)
state_dict = checkpoint["state_dict"]
for key in list(state_dict.keys()) :
    if key.startswith("model"):
        state_dict[key[6:]] = state_dict[key]
        state_dict.pop(key)

# model = T5ForConditionalGeneration.from_pretrained(model_path)
# tokenizer = JieBaTokenizer.from_pretrained(model_path)
model = CPTForConditionalGeneration.from_pretrained(model_path, ignore_mismatched_sizes=True)
tokenizer = CPTTokenizer.from_pretrained(model_path, ignore_mismatched_sizes=True)
#tokenizer = LlamaTokenizer.from_pretrained(model_path, add_eos_token=True)
#model = AutoModelForCausalLM.from_pretrained(model_path)

model.load_state_dict(state_dict)
while True:
    text = input("输入：") 
    ids = tokenizer.encode(text, return_tensors='pt')
    output = model.generate(ids,
                        decoder_start_token_id=tokenizer.cls_token_id,
                        eos_token_id=tokenizer.sep_token_id,
                        max_length=1024).numpy()[0]
    print(''.join(tokenizer.decode(output[1:])).replace(' ', ''))
