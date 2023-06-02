# -*- coding: utf-8 -*-
# @Time    : 2023/3/29 11:25
import numpy as np
from sacrebleu.metrics import BLEU
# from rouge import Rouge
from rouge_chinese import Rouge
import jieba


def evaluate(data):
    bleu_scorer_obj = BLEU(tokenize='zh')
    rouge_scorer_obj = Rouge()
    bleu_score = []
    for d in data:
        score = bleu_scorer_obj.sentence_score(
            hypothesis=d['text'],
            references=d['ref'],
        )
        bleu_score.append(score.score)

    bleu_score_avg = np.average(np.asarray(bleu_score))

    rouge_score = []
    for d in data:
        score = rouge_scorer_obj.get_scores(
            hyps=' '.join(jieba.cut(d['text'])),
            # hyps=' '.join(d['text']),
            refs=' '.join(jieba.cut(d['ref'][0])),
            # refs=' '.join(d['ref'][0]),
        )
        rouge_score.append(score[0]["rouge-l"]["f"])

    rouge_score_avg = np.average(np.asarray(rouge_score))

    return {
        "bleu_score": bleu_score,
        "bleu_score_avg": bleu_score_avg,
        "rouge-l_score": rouge_score,
        "rouge-l_score_avg": rouge_score_avg
    }


if __name__ == '__main__':
    import json
    eval_file_name = 'sft-epoch_9-dialog4_aug.json'
    eval_file_dir = 'output_loan_alpaca-dialog4_aug_v3'
    eval_file_path = f'/home/fm001/wangyuxuan/data/loan/{eval_file_dir}/{eval_file_name}'
    with open(eval_file_path, mode='r', encoding='utf-8') as f:
        list_data_dict = f.readlines()
    list_data_dict = [json.loads(l) for l in list_data_dict]
    eval_data = [
        {
            "text": data_dict['output_sft'],
            "ref": [data_dict['output']]
        } for data_dict in list_data_dict
    ]

    result = evaluate(eval_data)
    print('BLEU: ', result['bleu_score_avg'], 'Rouge-l: ',result['rouge-l_score_avg'])
