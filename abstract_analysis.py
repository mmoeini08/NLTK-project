# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 20:19:14 2022

@author: mmoein2
"""

#Libraries needed to run the tool
import numpy as np
from nltk import word_tokenize
from nltk.book import gutenberg
from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
import os
import nltk
from nltk.text import Text  
from nltk.probability import FreqDist
sns.set(style='darkgrid')

print("")
with open('abs.txt','r') as file:
    abstract = file.read()
# print(abstract)
###Searching text

# print(word_tokenize(abstract))
# print(sent_tokenize(abstract))

#concordance - where text was used
tokens = word_tokenize(abstract)
abs1 = Text(tokens)
print(abs1.concordance("model"))
print("")

#similar - show similar words depending on the text
print("Similar to 'model' in abstract:")
print(abs1.similar("model"))
print("")


#dispersion_plot - plot when a word is used in a text
abs1.dispersion_plot(["optimization", "ensemble", "physical"])
#could not find how to save figure


#Compile frequency distribution
fdist1 = FreqDist(abs1)

#Show that the result is a fdist is a class
print("")
print("Show frequency distribution of abstract:")
print(fdist1)
print("")
print("Typology of variable 'fdist1': {0}".format(type(fdist1)))
print("")


#Show number of times a single word is present in a text
print("Number of times the word 'model' is used in abstract: {0}".format(fdist1["model"]))
print("")

#Plot the frequency distribution
fdist1.plot(50) #, cumulative=True)

