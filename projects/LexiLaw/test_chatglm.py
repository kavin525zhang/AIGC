from transformers import AutoTokenizer, AutoModel
model_path = "/data9/NFS/patent/model_hub/chatglm2-6b/"
#model_path = "/home/zhangkaiming/projects/private_project/TensorFly/MedicalGPT/final_output"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# chatglm1
#model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
# chatglm2
model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device="cuda")
model = model.eval()
text= "可以申请撤销监护人的监护资格?"
response, history = model.chat(tokenizer, text, history=[], max_new_tokens=2048)
print(response)
