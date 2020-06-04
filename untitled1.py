# -*- coding: utf-8 -*-
"""
Created on Sat May  2 19:23:13 2020

@author: Boishakhi
"""



import nltk
#from nltk.stream.lancaster import LancasterStreamer
#streamer = LancasterStreamer()

import numpy
import tflearn
import tensorflow 
import random 
import json 

with open(r'F:\PROJECTS\chatbot\codes\dataset.json') as f: 
    data = json.load(f)

'''words = []
labels = []
docs = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs.append(pattern)
        
    if intent["tag"] not in lables:
        labels.append(intent["tag"])'''
        
       