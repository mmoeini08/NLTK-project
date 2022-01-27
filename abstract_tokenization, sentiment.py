# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 20:19:14 2022

@author: mmoein2
"""

#Libraries needed to run the tool
import numpy as np
from nltk import *
#from nltk.book import *
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')

print("")

with open('abs.txt','r') as file:
    abstract = file.read()
###Creating text
sentence = "CME594 Introduction to Data Science is a fun course."

#Create text in the format of TextBlob
sentTB = TextBlob(sentence)


#Show the grammatical meaning of each word
print("Tags of sentence: {0}".format(sentTB.tags))
print("")

#Show only nouns
print("Nouns phrases in sentence: {0}".format(sentTB.noun_phrases))
print("")

#Sentiment Analysis
print("Sentiment of sentence: {0}".format(sentTB.sentiment))
print("Polarity of sentence: {0}".format(sentTB.sentiment.polarity))
print("Subjectivity of sentence: {0}".format(sentTB.sentiment.subjectivity))
print("")

#Tokenization
print("Number of characters in sentence: {0}".format(len(sentTB)))
print("Number of words in sentence: {0}".format(len(sentTB.words)))
print("List of words in sentence: {0}".format(sentTB.words))
print("")


#Create paragraph
text = "CME594 Introduction to Data Science is the most interesting and fun course. We learn machine learning algorithms. It is taught by Sybil, who is funny. Even though it requires a lot of work, I am very interested by it."
textTB = TextBlob(text) #put the paragraph in TextBlob format

#Tokenization
print("Number of characters in paragraph {0}".format(len(text)))
print("Number of words in paragraph: {0}".format(len(textTB.words)))
print("Number of sentences in paragraph: {0}".format(len(textTB.sentences)))
print("")

#Sentiment Analysis
print("Sentiment of paragraph: {0}".format(textTB.sentiment))
print("Sentiment of each sentence in paragraph:")
print("")
for sent in textTB.sentences:
    print("Sentiment of sentence '{0}':".format(sent))
    print(sent.sentiment)
    print("")
