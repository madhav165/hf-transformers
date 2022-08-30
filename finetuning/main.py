import pandas as pd
import os
from torch import nn
os.environ['TRANSFORMERS_CACHE'] = os.path.join(os.getcwd(), str('/'.join(['models','pretrained'])))
os.environ['HF_HOME'] = os.path.join(os.getcwd(), str('/'.join(['datasets','prebuilt'])))

from transformers import AutoTokenizer, DataCollatorForLanguageModeling, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
import torch
from torch import nn

df_train = pd.read_csv(os.path.join('finetuning', 'train.csv'))
parent_codes = df_train.loc[df_train['Parent'].isna()]['Code.1']
chapter_codes = df_train.loc[df_train['Parent.1'].isin(parent_codes)]['Code.1']
major_codes = df_train.loc[df_train['Parent.1'].isin(chapter_codes)]['Code.1']
df_train = df_train.loc[~((df_train['Code.1'].isin(parent_codes))|(df_train['Code.1'].isin(chapter_codes))|(df_train['Code.1'].isin(major_codes)))]

y_train = df_train['Code.1'].apply(lambda x: str(x).replace(' ',''))
x_train = df_train['Self-explanatory texts']
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenized_x_train = tokenizer([' '.join(x) for x in x_train], truncation=True)
tokenized_y_train = tokenizer([' '.join(x) for x in y_train], truncation=True)
tokenized_x_train['labels'] = tokenized_y_train['input_ids']
# print(tokenized_x_train.keys())

block_size = 128
def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result

lm_dataset = Dataset.from_dict(tokenized_x_train)
# .map(group_texts, batched=True)

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

model = AutoModelForCausalLM.from_pretrained("distilgpt2")

training_args = TrainingArguments(
    output_dir="./finetuning/test_trainer",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset,
    eval_dataset=lm_dataset,
    data_collator=data_collator,
)

trainer.train()

pt_save_directory = "./finetuning/save_pretrained"

tokenizer.save_pretrained(pt_save_directory)
model.save_pretrained(pt_save_directory)

# from transformers import pipeline, set_seed
# generator = pipeline('text-generation', model=pt_save_directory)
# set_seed(42)
# x_train.iloc[0]
