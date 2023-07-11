from gensim.models import KeyedVectors
model = KeyedVectors.load_word2vec_format('./tencent-ailab-embedding-zh-d200-v0.2.0/tencent-ailab-embedding-zh-d200-v0.2.0.txt', binary=False)
print(model.similarity('碳捕捉', 'CO2捕捉'))
#model.most_similar(positive=['女','国王'],negative=['男'],topn=1)
#model.doesnt_match("上海 成都 广州 北京".split(" "))
print(model.most_similar('碳捕捉', topn=10))