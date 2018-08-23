#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 12:25:41 2018

@author: esraa_aldreabi
"""

import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer 
from sklearn.linear_model import LogisticRegression
from bs4 import BeautifulSoup

wordnet_lemmatizer = WordNetLemmatizer()# turn word to base form (dogs dog = dog)

stopwords = set(w.rstrip() for w in open('stopwords.txt'))

# load reviews
positive_reviews = BeautifulSoup(open('electronics/positive.review').read())
positive_reviews = positive_reviews.find_all('review_text')

negative_reviews = BeautifulSoup(open('electronics/negative.review').read())
negative_reviews = negative_reviews.find_all('review_text')

# the dataset has more positive than negative reviews 
#so I will shuffel the positive reviews and cut some 
#to make the dataset same size of positive and negative

np.random.shuffle(positive_reviews)
positive_reviews = positive_reviews[:len(negative_reviews)]




def tokenizer(s):
    s = s.lower()#lower case
    tokens = nltk.tokenize.word_tokenize(s) #split string to words
    tokens = [t for t in tokens if len(t) > 2] # keep words greater than 2 letters
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]#base form
    tokens = [t for t in tokens if t not in stopwords]# remove stopwords
    return tokens
#create index for each word in the final data vector( we need the size of vector)

word_index_map = {} # map word to index
current_index = 0

#save tokinized lists

positive_tokenized = []
negative_tokenized = []

    
for review in positive_reviews:
    tokens = tokenizer(review.text)
    positive_tokenized.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1
    
for review in negative_reviews:
    tokens = tokenizer(review.text)
    negative_tokenized.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1
# create data array for each token
def tokens_to_vector(tokens, label): # put tokens plus label 
    x = np.zeros(len(word_index_map) + 1) # vocabulary map +1 for label
    for t in tokens:
        i = word_index_map[t]
        x[i] += 1
    x = x / x.sum()
    x[-1] = label # last element label
    return x

N = len(positive_tokenized) + len(negative_tokenized)
data = np.zeros((N, len(word_index_map) + 1)) #initalize zeros
i = 0 # counter
for tokens in positive_tokenized:
    xy = tokens_to_vector(tokens, 1)
    data[i,:] = xy
    i += 1
for tokens in negative_tokenized:
    xy = tokens_to_vector(tokens, 0)
    data[i,:] = xy
    i += 1
np.random.shuffle(data)

X = data[:, :-1] #everything exept the last column
Y = data[:, -1]# the last column only 

Xtrain = X[:-100,]
Ytrain = Y[:-100,]
Xtest = X[-100:,]#last 100 test raws 
Ytest = Y[-100:,]

model = LogisticRegression()
model.fit(Xtrain, Ytrain)
print ("classification rate:", model.score(Xtest, Ytest))

threshold = 0.5
for word, index in iteritems(word_index_map):
    weight = model.coef_[0][index]
    if weight > threshold or weight < -threshold:
        print(word, weight)