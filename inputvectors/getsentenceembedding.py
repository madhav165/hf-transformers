import os
os.environ['TRANSFORMERS_CACHE'] = os.path.join(os.getcwd(), str('/'.join(['models','pretrained'])))
os.environ['HF_HOME'] = os.path.join(os.getcwd(), str('/'.join(['datasets','prebuilt'])))

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

class GetSentenceEmbedding(object):

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        # self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, sample):
        encoded_input = self.tokenizer([sample], padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        # out_state = self.model(**inp_encoded)
        # data = out_state
        return sentence_embeddings