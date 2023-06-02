from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments
from transformers import HfArgumentParser
from data_utils import train_info_args, NN_DataHelper
from models import MyTransformer, ChatGLMTokenizer, LoraArguments, setup_model_profile, ChatGLMConfig

if __name__ == '__main__':
    from utils.prompter import Prompter
    from alpaca2qa_loan_aug import process_profile

    prompter = Prompter('loan_template')

    train_info_args['seed'] = None
    parser = HfArgumentParser((ModelArguments, DataArguments,))
    model_args, data_args = parser.parse_dict(train_info_args, allow_extra_keys=True)

    setup_model_profile()

    dataHelper = NN_DataHelper(model_args, None, data_args)
    tokenizer: ChatGLMTokenizer
    tokenizer, config, _, _ = dataHelper.load_tokenizer_and_config(
        tokenizer_class_name=ChatGLMTokenizer, config_class_name=ChatGLMConfig)
    assert tokenizer.eos_token_id == 130005
    config.initializer_weight = False

    pl_model = MyTransformer(config=config, model_args=model_args)

    model = pl_model.get_llm_model()
    if not model.quantized:
        # 按需修改，目前只支持 4/8 bit 量化 ， 可以保存量化模型
        model.half().quantize(4).cuda()
    else:
        # 已经量化
        model.half().cuda()
    model = model.eval()

    while 1:
        prompt = input("输入内容:\n")
        prompt = prompt.replace('\\n', '\n').replace('\\"', '\"')
        instruction = prompt.split('#画像：') if '#画像：' in prompt else [prompt, '']
        instruction, input_ = instruction[0],instruction[1]
        _, p_dict_fea_map, p_concat = process_profile(input_)
        prompt = prompter.generate_prompt(instruction, p_concat)

        response, history = model.chat(tokenizer, prompt, history=[], max_length=2048,
                                       eos_token_id=config.eos_token_id,
                                       do_sample=True, top_p=0.7, temperature=0.95, )
        response = response.format_map(p_dict_fea_map)
        print(prompt, ' ', response)
