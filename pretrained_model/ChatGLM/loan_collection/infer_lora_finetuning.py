import os
from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments
from transformers import HfArgumentParser
from data_utils import train_info_args, NN_DataHelper
from models import MyTransformer, ChatGLMTokenizer, setup_model_profile, ChatGLMConfig, LoraArguments, global_args

if __name__ == '__main__':
    from utils.prompter import Prompter
    from alpaca2qa_loan_aug import process_profile

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

    assert lora_args.inference_mode == True and config.pre_seq_len is None
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

        while 1:
            prompt = input("输入内容:\n")
            prompt = prompt.replace('\\n', '\n').replace('\\"', '\"')
            instruction = prompt.split('#画像：') if '#画像：' in prompt else [prompt, '']
            instruction, input_ = instruction[0], instruction[1]
            _, p_dict_fea_map, p_concat = process_profile(input_)
            prompt = prompter.generate_prompt(instruction, p_concat)

            response, history = model.chat(tokenizer, prompt, history=[], max_length=2048,
                                           eos_token_id=config.eos_token_id,
                                           do_sample=True,
                                           num_beams=1,
                                           top_p=0.7, temperature=0.95, )
            response = response.format_map(p_dict_fea_map)
            print(prompt, ' ', response)
