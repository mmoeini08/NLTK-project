# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 20:19:14 2022

@author: mmoein2
"""
#Libraries needed to run the tool
import numpy as np
from nltk import *
from nltk.sentiment import SentimentIntensityAnalyzer
#from nltk.book import *
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')
from nltk.tokenize import sent_tokenize
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
print("")
with open('abs.txt','r') as file:
    abstract = file.read()
# tokens = sent_tokenize(abstract)
# abs1 = Text(tokens)
###Creating text
text = abstract
#Sometimes, it is good to turn all characters into lower case.
text_lower = text.lower()
print(text_lower)
print("")
#Work tokenizing the text
text_word_tokenize = word_tokenize(text)
print("Word tokenizing the text:")
print(text_word_tokenize)
print("")
#Sentence tokenizing the text
text_sentence_tokenize = sent_tokenize(text)
print("Sentence tokenizing the text:")
print(text_sentence_tokenize)
print("")
#Removing stop words (frequent words that do not add meaning like 'a')
stopwords = corpus.stopwords.words("english") #NLTK already has a list.
text_stopwords = [word for word in text_word_tokenize if word not in stopwords]
print("Removing stop words from the text:")
print(text_stopwords)
print("")
#Lemmatizing the text
wordnet_lemmatizer = WordNetLemmatizer()
text_lemmatized = [wordnet_lemmatizer.lemmatize(word) for word in text_stopwords]
print("Lemmatizing the text:")
print (text_lemmatized) #Notice the removal of 's' in 'algorithms'
print("")
#Stemming the text
word_stemmer = SnowballStemmer('english')
text_stemmed = [word_stemmer.stem(word) for word in text_stopwords]
print("Stemming the text:")
print (text_stemmed)
print("")
#Getting tags from the text (i.e., nouns, verbs, etc.)
text_tags = pos_tag(text_stopwords)
print(text_tags)
print("")
#Sentiment Analysis
sia = SentimentIntensityAnalyzer()
#Sentiment of the all text
text_sentiment = sia.polarity_scores(text)
print("Sentiment of the entire text:")
print(text_sentiment)
print("")
#Sentiment of each sentence
text_sentence_sentiment = [sia.polarity_scores(sentence) for sentence in text_sentence_tokenize]
i = 1
zdata=[]
xdata=[]
ydata=[]
for result in text_sentence_sentiment:
    print("Sentiment of sentence {0}:".format(i))
    i+=1
    print(result)
    print("")
for i in range(0,12):
    result=text_sentence_sentiment[i]
    for key, value in result.items():
        print(key, ' : ', value)
        if key=='neg':
            zdata.append(value)
        if key=='neu':
            xdata.append(value)
        if key=='pos':
            ydata.append(value)
ax = plt.axes(projection='3d')
ax.set_xlabel('neu',fontsize=20)
ax.set_ylabel('pos', fontsize=20)
ax.set_zlabel('neg',fontsize=20)
ax.scatter3D(zdata, xdata, ydata, cmap='Greens')
