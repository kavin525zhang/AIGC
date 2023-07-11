import json
import os


def load_json_from_file(file_name):
    with open(file_name, 'r', encoding='utf-8') as file:
        data = json.load(file)
        data_item = data['items']
    return data_item


def convert_to_target_format(data, start_id):
    result = []

    for entry in data:
        question = entry["instruction"].replace('\\n', '\n')
        input = entry['input'].replace('\\n', '\n')
        raw_answer = entry["output"].replace("\\n\\n", "\n \n").replace('\\n', '\n')
        answer = raw_answer.split("\n")
        result.append({"id": start_id, "paragraph": [{"q": question + '\n' + input, "a": answer}]})
        start_id += 1

    return result


def save_json_to_file(data, file_name):
    with open(file_name, 'w', encoding='utf-8') as file:
        for item in data:
            json_line = json.dumps(item, ensure_ascii=False)
            file.write(json_line + "\n")


input_folder = '/home/ubuntu/workspace/data/loan'
files = os.listdir(input_folder)

merged_data = []

for file in files:
    if file.endswith('.json'):
        file_path = os.path.join(input_folder, file)
        data = load_json_from_file(file_path)
        merged_data.extend(data)

start_id = 1
converted_data = convert_to_target_format(merged_data, start_id)

save_json_to_file(converted_data, os.path.join(input_folder, 'train.json'))
