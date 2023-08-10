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
from flask import Flask, redirect,url_for, request, render_template

# Load pre-trained BART tokenizer and model
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('./fine_tuned_model')


app=Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def Summarize():
    if request.method == 'POST':
        input_text = request.form['text']
        input_ids = tokenizer.encode(input_text, return_tensors='pt', max_length=256, truncation=True)

        # Generate a summary
        summary_ids = model.generate(input_ids, max_length=150, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        print(summary)
        return render_template('index.html', inputText = input_text, summary = summary)
    return render_template('index.html')

app.run(debug=False, port=80)
