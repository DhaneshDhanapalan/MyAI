# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 11:45:56 2020

@author: Dhanesh
"""

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation

f=open('D:/Tattvamasi/NLP/Project_Impelsys/dataset/dataset/stories_text_summarization_dataset_train/000c835555db62e319854d9f8912061cdca1893e.STORY','r',errors = 'ignore')
text = f.read()

#print(text)

stopwords = list(STOP_WORDS)
#print(stopwords)
nlp = spacy.load('en_core_web_sm')
doc = nlp(text)

tokens = [token.text for token in doc]
#print(tokens)

#print(punctuation)

#text cleaning is removing stop words and punctuation
#Here we are removing stopwors and punctuation

word_frequencies = {}
for word in doc:
    if word.text.lower() not in stopwords:
        if word.text.lower() not in punctuation :
            if word.text not in word_frequencies.keys():
                word_frequencies[word.text] = 1
            else:
                word_frequencies[word.text] += 1
                
#print(word_frequencies)

max_frequency = max(word_frequencies.values())
#print(max_frequency)


#to get the normalized frequency, divide each frequency with max frequency
#The maximum value of normalized frequency will be one.

for word in word_frequencies.keys():
    word_frequencies[word] = word_frequencies[word]/max_frequency
    
#print(word_frequencies)

#senetence tokenization

sentence_tokens = [sent for sent in doc.sents]
#print(sentence_tokens)

#calculating the sentence score for each sentence same way as word frequency.
#We have normalized frequency for every word
#We are adding the normalized frequency here
#And getting the important sentences
sentence_scores = {}
for sent in sentence_tokens:
    for word in sent:
        if word.text.lower() in word_frequencies.keys():
            if sent not in sentence_scores.keys():
                sentence_scores[sent] = word_frequencies[word.text.lower()]
            else:
                sentence_scores[sent] += word_frequencies[word.text.lower()]
                
#print(sentence_scores)
#now we need to get 10% of sentence with maximumm score and is done by heapq

from heapq import nlargest

select_length = int(len(sentence_tokens)*0.1)
#print(select_length)

summary = nlargest(select_length, sentence_scores, key = sentence_scores.get)

#print(summary)
final_summary = [word.text for word in summary]
summary = ''.join(final_summary)

print(summary)

print(len(text))
print(len(summary))