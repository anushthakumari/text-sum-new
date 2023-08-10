#!/usr/bin/env python
# coding: utf-8


import json
import re
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import json
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import pipeline
from transformers import BartTokenizer, BartForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AdamW


nltk.download('punkt')


# In[15]:


with open('./AI_intro.json', 'r') as json_file:
    dataset = json.load(json_file)

# Define a function to preprocess text
def preprocess_text(text):
    if isinstance(text, str):  # Check if the value is a string
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
    return text


model_name = 'bert-base-uncased'
model = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)


# Load the JSON file
with open('./AI_intro.json', 'r') as json_file:
    data = json.load(json_file)

preprocessed_data = {}
for key, value in data.items():
    if key == 'text':
        # Tokenize and encode the text
        encoding = tokenizer.encode_plus(
            value,
            add_special_tokens=True,
            max_length=256,
            padding='max_length',
            return_tensors='pt',
            return_attention_mask=True
        )
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
    else:
        # For other keys, simply copy the value
        preprocessed_data[key] = value

# Save the preprocessed data to a new JSON file
output_file = 'preprocessed_tokenized_data.json'
with open(output_file, 'w') as json_file:
    json.dump(preprocessed_data, json_file, indent=2)

print(f'Preprocessed and tokenized data saved to {output_file}')


# Load pre-trained BART tokenizer and model
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

with open('preprocessed_tokenized_data.json', 'r') as json_file:
    preprocessed_data = json.load(json_file)

inputs = []
for text in preprocessed_data:
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=300,
        padding='max_length',
        return_tensors='pt',
        return_attention_mask=True
    )
    inputs.append(encoding)

inputs = tokenizer(text, padding='max_length', truncation=True, return_tensors='pt', max_length=256)

# Convert list of encodings to tensors
input_ids = torch.cat([inputs['input_ids']], dim=0)
attention_mask = torch.cat([inputs['attention_mask']], dim=0)

input_ids = torch.cat([inputs['input_ids']], dim=0)
attention_mask = torch.cat([inputs['attention_mask']], dim=0)

train_inputs = input_ids
train_masks = attention_mask

class SummarizationDataset(Dataset):
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx]
        }

# Create dataloader
dataset = SummarizationDataset(input_ids, attention_mask)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Define optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=1e-5)

# Fine-tuning loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc='Epoch {}'.format(epoch + 1)):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(dataloader)
    print(f'Epoch {epoch + 1}, Average Loss: {average_loss:.4f}')

# Save the fine-tuned model
model.save_pretrained('./fine_tuned_model')