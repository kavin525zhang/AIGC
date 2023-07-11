import json

import re
import random

random.seed(42)

from utils.prompter import Prompter

prompter = Prompter('loan_template')

features = [
    '性别',
    '债务人姓名',
    '称谓',
    '身份',
    '欠款类型',
    '逾期时长',
    '信用卡尾号',
    '账单金额',
    '最低还款额',
    '工作单位',
    '户籍地',
    '管辖派出所',
]

common_input = ['贷款类型：信用卡', '']
gender_title_map = {'男': '先生', '女': '女士'}


def process_profile(profile, loan_type='信用卡'):
    if profile in common_input:
        p_dict = {f: '' for f in features}
    else:
        p_split = profile.replace('\"', '').split('\n')
        p_dict = {content.split('：')[0]: content.split('：')[1] for content in p_split}
        p_dict = {v: p_dict.get(v, '') for v in features}
    p_dict['称谓'] = gender_title_map.get(p_dict['性别'], '')
    p_dict['欠款类型'] = loan_type
    p_dict_fea_map = {k: v if v != '' else '{' + k + '}' for k, v in p_dict.items()}
    p_concat = '\n'.join(['：'.join([k, v]) for k, v in p_dict.items()])
    return p_dict, p_dict_fea_map, p_concat


def alaca2qa(src, dst, feature_map=True):
    global common_input

    with open(src, mode='r', encoding='utf-8') as f:
        list_data_dict = json.loads(f.read())['items']

    input_profile = set([e['input'] for e in list_data_dict if e['input'] not in common_input])
    sources = []
    targets_aug = []

    for example in list_data_dict:
        tmp_sources = []
        tmp_targets = []

        profile_dict, profile_fea_map, profile_concat = process_profile(example['input'])
        tmp_sources.append(prompter.generate_prompt(example['instruction'], profile_concat))
        # replace the feature map into real feature value
        example_output = example['output'].format_map(profile_fea_map) if feature_map else example['output']
        tmp_targets.append(example_output)

        # special case
        if example['instruction'] == '中国的计划单列市有哪些？':
            sources.extend(tmp_sources)
            targets_aug.extend(tmp_targets)
            continue

        # random sample profile input for data augmentation
        if_common_input = True if example['input'] in common_input else False
        random_num = 4 if if_common_input else 3
        tmp_profile = [p for p in input_profile if p != example['input']]
        random_profile = random.sample(tmp_profile, random_num)
        if not if_common_input:
            random_profile.append('')

        for p in random_profile:
            _, p_dict_fea_map, p_concat = process_profile(p)
            if feature_map:
                output = example['output'].format_map(p_dict_fea_map)
            else:
                output = example['output']
            tmp_sources.append(prompter.generate_prompt(example['instruction'], p_concat))
            tmp_targets.append(output)
        sources.extend(tmp_sources)
        targets_aug.extend(tmp_targets)

    assert len(sources) == len(targets_aug)

    with open(dst, mode='w', encoding='utf-8', newline='\n') as f:

        for i, (s, t) in enumerate(zip(sources, targets_aug)):
            paragraph = [
                {
                    'q': s,
                    'a': [t]
                }
            ]
            f.write(json.dumps({'id': i + 1, 'paragraph': paragraph}, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    import os

    file_name = 'dialog4_aug'
    feature_map = True
    file_version = 'v3' if feature_map else 'v2'

    data_path = '/home/fm001/wangyuxuan/data/loan'
    src = os.path.join(data_path, f'{file_name}.json')
    dst = os.path.join(data_path, f'alpaca-{file_name}_{file_version}.json')
    alaca2qa(src, dst, feature_map=feature_map)
