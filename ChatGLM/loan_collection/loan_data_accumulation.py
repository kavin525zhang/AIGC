import json


def accumulate_data(src, dst, combo_name='dialog_all'):
    with open(src, mode='r', encoding='utf-8') as f:
        src_data_dict = json.loads(f.read())['items']

    with open(dst, mode='r', encoding='utf-8') as f:
        dst_data_dict = json.loads(f.read())['items']

    src_data_dict += dst_data_dict

    instruction_to_track = []
    src_data_to_keep = []

    for combo in src_data_dict:
        if combo['instruction'] not in instruction_to_track:
            instruction_to_track.append(combo['instruction'])
            src_data_to_keep.append(combo)
    assert len(instruction_to_track) == len(src_data_to_keep)

    global data_path
    with open(os.path.join(data_path, combo_name + '.json'), mode='w', encoding='utf-8') as f:
        f.write(json.dumps({'items': src_data_to_keep}, ensure_ascii=False))


if __name__ == '__main__':
    import os

    data_path = '/home/fm001/wangyuxuan/data/loan'

    src = os.path.join(data_path, r'dialog3_aug.json')
    dst = os.path.join(data_path, r'dialog4_toadd.json')

    accumulate_data(src, dst, 'dialog4_aug')
