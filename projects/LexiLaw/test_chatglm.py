from transformers import AutoTokenizer, AutoModel
model_path = "/data9/NFS/patent/model_hub/chatglm-6b/"
#model_path = "/home/zhangkaiming/projects/private_project/TensorFly/MedicalGPT/final_output"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# chatglm1
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
# chatglm2
#model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device="cuda")
model = model.eval()
text= "你现在是一个精通中国法律的法官，请对以下案件做出分析:经审理查明：被告人xxx于2017年12月，多次在本市xxx盗窃财物。具体事实如下：（一 ）2017年12月9日15时许，被告人xxx在xxx店内，盗窃白色毛衣一件（价值人民币259元）。现赃物已起获并发还。（二）2017年12月9日16时许，被告人xx在本>市xxx店内，盗窃米白色大衣一件（价值人民币1199元）。现赃物已起获并发还。（三）2017年12月11日19时许，被告人xxx在本市xxx内，盗窃耳机、手套、化>妆镜等商品共八件（共计价值人民币357.3元）。现赃物已起获并发还。（四）2017年12月11日20时许，被告人xx在本市xxxx内，盗窃橙汁、牛肉干等商品共四>件（共计价值人民币58.39元）。现赃物已起获并发还。2017年12月11日，被告人xx被公��机关抓获，其到案后如实供述了上述犯罪事实。经鉴定，被告人xxx被 诊断为精神分裂症，限制刑事责任能力，有受审能力。"
response, history = model.chat(tokenizer, text, history=[], max_new_tokens=2048)
print(response)
