import os

device_id = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = device_id

from fastapi import FastAPI, Request
import uvicorn, json, datetime
import torch
from loguru import logger

logger.add('log/runtime_{time}.log')

from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments
from deep_training.nlp.models.chatglm import setup_model_profile, ChatGLMConfig
from deep_training.nlp.models.lora.v2 import LoraArguments
from transformers import HfArgumentParser

from data_utils import train_info_args, NN_DataHelper
from models import MyTransformer, ChatGLMTokenizer, InvalidScoreLogitsProcessor, LogitsProcessorList
from tqdm import trange
from utils.prompter import Prompter
from alpaca2qa_loan_aug import process_profile

prompter = Prompter('loan_template')

DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


app = FastAPI()


@app.post("/")
async def create_item(request: Request):
    global model, tokenizer, features
    json_post_raw = await request.json()
    logger.info('request:')
    logger.info(json_post_raw)
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    messages = json_post_list.get('messages')

    if len(messages) == 0:
        logger.error('request messages is empty!')
        return {'status': 500, 'info': 'request messages is empty!'}
    if messages[-1]['role'] != 'user':
        logger.error('request messages must end with user!')
        return {'status': 500, 'info': 'request messages must end with user!'}
    if messages[0]['role'] != 'user':
        messages = messages[1:]

    # query = ''
    # i = 0
    # for message in messages:
    #     if i % 2 == 0:
    #         query += f'[Round {str(i // 2 + 1)}]\n'
    #     role = message['role']
    #     content = message['content']
    #     if role == 'user':
    #         query += f'问:{content}\n'
    #     else:
    #         query += f'答:{content}\n'
    #     i += 1
    # query += '答:'
    # logger.info('query:')
    # logger.info('\n' + query)

    message = messages[-1]
    prompt = message['content']
    prompt = prompt.replace('\\n', '\n').replace('\\"', '\"')
    logger.info('query:')
    logger.info('\n' + prompt)

    instruction = prompt.split('#画像：') if '#画像：' in prompt else [prompt, '']
    instruction, input_ = instruction[0], instruction[1]
    _, p_dict_fea_map, p_concat = process_profile(input_)
    prompt = prompter.generate_prompt(instruction, p_concat)

    history = json_post_list.get('history')
    max_length = json_post_list.get('max_length')
    top_p = json_post_list.get('top_p')
    temperature = json_post_list.get('temperature')
    num_beams = json_post_list.get('num_beams', 1)

    response, history = model.chat(tokenizer,
                                   prompt,
                                   history=history,
                                   num_beams=num_beams,
                                   max_length=max_length if max_length else 2048,
                                   top_p=top_p if top_p else 0.7,
                                   temperature=temperature if temperature else 0.95)
    response = response.format_map(p_dict_fea_map)

    logger.info('response:')
    logger.info('\n' + response)

    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        "history": history,
        "status": 200,
        "time": time
    }
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    print(log)
    torch_gc()
    return answer


@app.post("/batch")
async def create_item_batch(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()
    logger.info('request:')
    logger.info(json_post_raw)
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    list_data_dict = json_post_list.get('items')

    if len(list_data_dict) == 0:
        logger.error('request items is empty!')
        return {'status': 500, 'info': 'request items is empty!'}

    profile_list = [process_profile(example['input']) for example in list_data_dict]
    prompt_list = [prompter.generate_prompt(example['instruction'], profile_list[idx][-1]) for idx, example
                   in enumerate(list_data_dict)]

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
    response_list = []

    global_batch = 10

    for i in trange(0, len(prompt_list), global_batch):
        tmp_prompt_list = prompt_list[i:i + global_batch]
        tmp_profile_list = profile_list[i:i + global_batch]

        # inference by batch
        inputs = tokenizer(tmp_prompt_list, return_tensors="pt", padding=True)
        inputs = inputs.to(model.device)
        outputs = model.generate(**inputs, **gen_kwargs)
        tmp_response_list = [model.process_response(tokenizer.decode(output[len(inputs["input_ids"][0]):])) for output
                             in outputs.tolist()]
        tmp_response_list = [response.format_map(tmp_profile_list[idx][1]) for idx, response in
                             enumerate(tmp_response_list)]
        response_list += tmp_response_list
        # torch_gc()
    assert len(prompt_list) == len(response_list)

    # update response
    for idx, example in enumerate(list_data_dict):
        example.update({
            "output_sft": response_list[idx]
        })
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "items": list_data_dict,
        "status": 200,
        "time": time
    }
    log = "[" + time + "]"
    print(log)
    torch_gc()
    return answer


if __name__ == '__main__':
    train_info_args['seed'] = None
    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_dict(train_info_args,allow_extra_keys=True)

    setup_model_profile()

    dataHelper = NN_DataHelper(model_args, None, data_args)
    tokenizer: ChatGLMTokenizer
    tokenizer, _, _, _ = dataHelper.load_tokenizer_and_config(
        tokenizer_class_name=ChatGLMTokenizer, config_class_name=ChatGLMConfig)

    model_name = 'best_ckpt_epoch_9'
    model_path = f'./output_loan_alpaca-dialog4_aug_v3/{model_name}'

    config = ChatGLMConfig.from_pretrained(model_path)
    config.initializer_weight = False

    lora_args = LoraArguments.from_pretrained(model_path)

    assert lora_args.inference_mode is True and config.pre_seq_len is None

    pl_model = MyTransformer(config=config, model_args=model_args, lora_args=lora_args,
                             # load_in_8bit=global_args["load_in_8bit"],
                             # # device_map="auto",
                             # device_map={"": 0},  # 第一块卡
                             )
    # 加载lora权重
    pl_model.load_sft_weight(model_path)

    model = pl_model.get_llm_model()
    # 按需修改
    model.half().cuda()
    model = model.eval()

    uvicorn.run(app, host='0.0.0.0', port=8505, workers=1)
