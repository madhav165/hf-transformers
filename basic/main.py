import os
from torch import nn
os.environ['TRANSFORMERS_CACHE'] = os.path.join(os.getcwd(), str('/'.join(['models','pretrained'])))
os.environ['HF_HOME'] = os.path.join(os.getcwd(), str('/'.join(['datasets','prebuilt'])))

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset, Audio

# Classifier

classifier = pipeline("sentiment-analysis")
result = classifier("We are very happy to show you the 🤗 Transformers library.")
print(f"label: {result[0]['label']}, with score: {round(result[0]['score'], 4)}")

results = classifier(["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."])
for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")

# Audio

speech_recognizer = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")
dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")
dataset = dataset.cast_column("audio", Audio(sampling_rate=speech_recognizer.feature_extractor.sampling_rate))
result = [speech_recognizer(x["array"]) for x in dataset[:4]["audio"]]
print([d["text"] for d in result])

# Multi lingual

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
result = classifier("Nous sommes très heureux de vous présenter la bibliothèque 🤗 Transformers.")
print(f"label: {result[0]['label']}, with score: {round(result[0]['score'], 4)}")

# Automodel

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
pt_batch = tokenizer(
    ["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."],
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt",
)
pt_model = AutoModelForSequenceClassification.from_pretrained(model_name)
pt_outputs = pt_model(**pt_batch)
pt_predictions = nn.functional.softmax(pt_outputs.logits, dim=-1)
print(pt_predictions)

# Save model

pt_save_directory = "./pt_save_pretrained"
tokenizer.save_pretrained(pt_save_directory)
pt_model.save_pretrained(pt_save_directory)
pt_model = AutoModelForSequenceClassification.from_pretrained("./pt_save_pretrained")