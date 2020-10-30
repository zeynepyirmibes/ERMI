# -*- coding: utf-8 -*-

import os
import io
import pandas as pd
from gensim.models import FastText
from datetime import datetime
from gensim.models.fasttext import FastText as FT_gensim
from gensim.test.utils import datapath

raw_path = "Fasttext"
language = "EL"
files_path = os.path.join(raw_path, language, "files.txt")

files = [line.strip().replace(".xz", "") for line in open(files_path, "r").readlines()]

def read_corpus(path):
    corpus_file = io.open(path, "r", encoding="utf-8")
    corpus = []
    for s in corpus_file:
        if not s.startswith('#'):
            corpus.append(s)
    corpus = [x.split('\t') for x in corpus]
    new_corpus = pd.DataFrame(corpus,
                              columns=['ID', 'FORM', 'LEMMA', 'UPOS', 'XPOS', 'FEATS', 'HEAD', 'DEPREL',
                                       'DEPS', 'MISC'])
    return new_corpus

def prepare_corpus(corpus):
    sentences = []
    dont_empty = False
    for index, row in corpus.iterrows():
      id = row['ID'].strip()
      if id[:2] == '1-':
          sentence = []
          sentence.append(row)
          dont_empty = True
      elif id == '1':
          if dont_empty:
            sentence.append(row)
            dont_empty = False
          else:
            sentence = []
            sentence.append(row)
      elif row['ID'] == '\n':
          sentences.append(sentence)
      else:
          sentence.append(row)
    return sentences

def multiple_word_indices(sentence):
  indices = []
  for word in sentence:
    id = word['ID'].strip()
    if '-' in id:
      first_ind, second_ind = id.split('-')
      first_ind = int(first_ind)
      second_ind = int(second_ind)
      for i in range(first_ind, second_ind+1):
        indices.append(i)
  return indices

def sentence_extractor(sentence):
  indices = multiple_word_indices(sentence)
  text = ''
  tokens = []
  for word in sentence:
    if '-' in word['ID']:
        text += word['FORM'] + ' '
        tokens.append(word['FORM'])
    elif int(word['ID'].strip()) not in indices:
        text += word['FORM'] + ' '
        tokens.append(word['FORM'])
  return text, tokens

def extract_sentences(sentences):
  text_list = []
  token_list = []
  for sentence in sentences:
    text, tokens = sentence_extractor(sentence)
    text_list.append(text)
    token_list.append(tokens)
  return text_list, token_list

raw_entire_token_list = []

for filename in files:
  dateTimeObj = datetime.now()
  print(dateTimeObj," Reading ", filename)
  raw_file_path = os.path.join(raw_path, language, filename)
  raw_corpus = read_corpus(raw_file_path)
  raw_sentences = prepare_corpus(raw_corpus)
  raw_text_list, raw_token_list = extract_sentences(raw_sentences)
  raw_entire_token_list.append(raw_token_list)

fasttext_tokens = []
for token_list in raw_entire_token_list:
  fasttext_tokens += token_list

del raw_entire_token_list

model_gensim = FT_gensim(size=300)

# build the vocabulary
dateTimeObj = datetime.now()
print(dateTimeObj," Building vocab ")
model_gensim.build_vocab(sentences=fasttext_tokens)

print(model_gensim)

# train the model
dateTimeObj = datetime.now()
print(dateTimeObj," Training gensim model ")
model_gensim.train(
    sentences=fasttext_tokens, epochs=model_gensim.epochs,
    total_examples=model_gensim.corpus_count, total_words=model_gensim.corpus_total_words
)

dateTimeObj = datetime.now()
print(dateTimeObj,"Successful Training of gensim model ")
print(model_gensim)

# saving a model trained via Gensim's fastText implementation
model_gensim.save("Fasttext/Embeddings/EL/gensim_el")
