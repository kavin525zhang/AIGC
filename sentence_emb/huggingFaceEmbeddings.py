import os
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms.base import LLM
import codecs
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sklearn.metrics.pairwise import cosine_similarity

os.environ["HUGGINGFACEHUB_API_TOKEN"] = 'HUGGINGFACEHUB_API_TOKEN'

texts = ["碳捕获", "一种碳捕捉装置", "热电厂具有CO2封存", "热电联产设备和热电联产方法", "碳捕集喷气发动机", "一种CO2捕捉收集系统"]
#texts = ["碳捕获", "燃气涡轮机" ,"燃烧", "热回收" ,"蒸汽涡轮机" ,"燃烧室"]
# select embeddings
#huggie = HuggingFaceEmbeddings(model_name="/data9/NFS/patent/model_hub/chatglm2-6b/")
huggie = HuggingFaceEmbeddings(model_name="/home/zhangkaiming/models/luotuo-bert-medium")
#huggie = HuggingFaceEmbeddings(model_name="/home/zhangkaiming/models/text2vec-large-chinese")
# print(texts)
# create vectorstores
# embeddings = huggie.embed_query(texts)
embeddings = huggie.embed_documents(texts)

s = cosine_similarity(embeddings)
print(s)