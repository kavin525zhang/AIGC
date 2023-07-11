import os
from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments
from transformers import HfArgumentParser
from data_utils import train_info_args, NN_DataHelper
from models import MyTransformer, ChatGLMTokenizer, setup_model_profile, ChatGLMConfig, LoraArguments, global_args, \
    InvalidScoreLogitsProcessor, LogitsProcessorList

if __name__ == '__main__':
    import json
    from utils.prompter import Prompter
    from tqdm import trange, tqdm

    from alpaca2qa_loan_aug import process_profile

    file_for_inference = 'dialog4_aug.json'
    data_path = '/home/fm001/wangyuxuan/data/loan'
    prompter = Prompter('loan_template')

    train_info_args['seed'] = None
    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_dict(train_info_args, allow_extra_keys=True)
    setup_model_profile()
    dataHelper = NN_DataHelper(model_args, None, data_args)
    tokenizer: ChatGLMTokenizer
    tokenizer, _, _, _ = dataHelper.load_tokenizer_and_config(
        tokenizer_class_name=ChatGLMTokenizer, config_class_name=ChatGLMConfig)

    ckpt_name = 'epoch_9'
    ckpt_path = 'output_loan_alpaca-dialog4_aug_v3'
    ckpt_dir = f'./{ckpt_path}/best_ckpt_{ckpt_name}'

    config = ChatGLMConfig.from_pretrained(ckpt_dir)
    config.initializer_weight = False
    lora_args = LoraArguments.from_pretrained(ckpt_dir)

    assert lora_args.inference_mode is True and config.pre_seq_len is None
    pl_model = MyTransformer(config=config, model_args=model_args, lora_args=lora_args,
                             # load_in_8bit=global_args["load_in_8bit"],
                             # # device_map="auto",
                             # device_map = {"":0} # 第一块卡
                             )
    # 加载lora权重
    pl_model.load_sft_weight(ckpt_dir)

    if getattr(pl_model.get_llm_model(), "is_loaded_in_8bit", False):
        pl_model.eval().cuda()
    else:
        pl_model.eval().half().cuda()

    enable_merge_weight = False
    if enable_merge_weight:
        # 合并lora 权重 保存
        pl_model.save_sft_weight(os.path.join(ckpt_dir, 'pytorch_model_merge.bin'), merge_lora_weight=True)

    else:
        model = pl_model.get_llm_model()

        # prepare data
        with open(os.path.join(data_path, file_for_inference), mode='r', encoding='utf-8') as f:
            list_data_dict = json.loads(f.read())['items']
        for example in list_data_dict:
            example['input'] = process_profile(example['input'])[2]
        prompt_list = [prompter.generate_prompt(example['instruction'], example['input']) for example in list_data_dict]

        # set params
        max_length = 2048
        num_beams = 1
        do_sample = True
        top_p = 0.7
        temperature = 0.95
        logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
                      "temperature": temperature, "logits_processor": logits_processor}

        # inference by batch
        response_list = []
        global_batch_size = 50

        for i in trange(0, len(prompt_list), global_batch_size):
            tmp_prompt_list = prompt_list[i:i + global_batch_size]
            inputs = tokenizer(tmp_prompt_list, return_tensors="pt", padding=True)
            inputs = inputs.to(model.device)
            outputs = model.generate(**inputs, **gen_kwargs)
            response_list.extend(
                [model.process_response(tokenizer.decode(output[len(inputs["input_ids"][0]):])) for output in
                 outputs.tolist()])
        assert len(prompt_list) == len(response_list)

        # update response
        for idx, example in tqdm(enumerate(list_data_dict)):
            example.update({
                "output_sft": response_list[idx]
            })
        # save file
        file_save_path = os.path.join(data_path, ckpt_path)
        if not os.path.exists(os.path.join(data_path, ckpt_path)):
            os.makedirs(file_save_path)
        with open(os.path.join(file_save_path, f"sft-{ckpt_name}-" + file_for_inference), mode='w', encoding='utf-8',
                  newline='\n') as f:
            for line in list_data_dict:
                f.write(json.dumps(line, ensure_ascii=False) + '\n')
