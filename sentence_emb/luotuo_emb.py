import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
from argparse import Namespace
# Import our models. The package will take care of downloading the models automatically
tokenizer = AutoTokenizer.from_pretrained("silk-road/luotuo-bert-medium")
model_args = Namespace(do_mlm=None, pooler_type="cls", temp=0.05, mlp_only_train=False, init_embeddings_model=None)
model = AutoModel.from_pretrained("silk-road/luotuo-bert-medium", trust_remote_code=True, model_args=model_args)

# Tokenize input texts
texts = [
    "词嵌入（Word embedding）是自然语言处理（NLP）中语言模型与表征学习技术的统称。概念上而言，它是指把一个维数为所有词的数量的高维空间嵌入到一个维数低得多的连续向量空间中，每个单词或词组被映射为实数域上的向量。",
    "词嵌入的方法包括人工神经网络、对词语同现矩阵降维、几率模型以及单词所在上下文的显式表示等。",
    "周杰伦出生于台湾省新北市，祖籍福建省泉州市永春县。4岁的时候，母亲叶惠美把他送到淡江山叶幼儿音乐班学习钢琴。",
    "初中二年级时，父母因性格不合离婚，他归母亲叶惠美抚养。中考时，没有考上普通高中，同年，因为擅长钢琴而被淡江中学第一届音乐班录取。高中毕业以后，两次报考台北大学音乐系均没有被录取，于是开始在一家餐馆打工。"
]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# Get the embeddings
with torch.no_grad():
    embeddings = model(**inputs, output_hidden_states=True, return_dict=True, sent_emb=True).pooler_output

# Calculate cosine similarities
# Cosine similarities are in [-1, 1]. Higher means more similar
print(torch.nn.functional.cosine_similarity(embeddings[0], embeddings[1], dim=0))
print(torch.nn.functional.cosine_similarity(embeddings[0], embeddings[2], dim=0))
print(torch.nn.functional.cosine_similarity(embeddings[0], embeddings[3], dim=0))
print(torch.nn.functional.cosine_similarity(embeddings[1], embeddings[2], dim=0))
print(torch.nn.functional.cosine_similarity(embeddings[1], embeddings[3], dim=0))
print(torch.nn.functional.cosine_similarity(embeddings[2], embeddings[3], dim=0))