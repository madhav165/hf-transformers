import os
os.environ['TRANSFORMERS_CACHE'] = os.path.join(os.getcwd(), str('/'.join(['models','pretrained'])))
os.environ['HF_HOME'] = os.path.join(os.getcwd(), str('/'.join(['datasets','prebuilt'])))

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class PassThroughTransformer(object):
    
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained("distilgpt2")

    def __call__(self, sample):
        inp_encoded = self.tokenizer([sample], padding=True, truncation=True, return_tensors='pt')
        out_state = self.model(**inp_encoded)
        logits = out_state.logits
        data = torch.argmax(logits, dim=-1)
        data = torch.nn.functional.pad(data, (0,100,0,0), "constant", 0)[:,1:101]
        data = data.type(torch.FloatTensor)
        return data